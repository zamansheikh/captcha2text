"""Export the attention model trained in captcha_reader.ipynb to ONNX for the server.

Run it in the environment the notebook trained in (TensorFlow 2.x with tf.keras), from the
folder that holds words_captcha/ and checkpoints/. Install the converter there first:

  pip install tf2onnx onnxruntime

  python export_attention.py \
      --annotations words_captcha/spec_train_val.txt \
      --checkpoint checkpoints/train \
      --out models/words-v1 \
      --eval 2000

The step-by-step decoding loop is built into the exported graph, so the server only has to
map the returned ids to letters. The tokenizer is rebuilt from the same annotation file the
notebook used, which gives the same letter-to-id mapping. --eval runs the exported ONNX
model on the notebook's validation images (lines after the first 100,000) and prints the
exact-match accuracy, which should be close to the notebook's 90.7%.

This file only needs TensorFlow, numpy, tf2onnx and onnxruntime; it does not import the
captcha2text package.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dropout, MaxPool2D

HEIGHT, WIDTH = 50, 100
EMBEDDING_DIM, UNITS = 256, 512
NUM_TRAIN = 100000


# ---- copied from captcha_reader.ipynb; must match the checkpoint exactly ----------------

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        layers = [
            Conv2D(96, (3, 3), padding='valid', activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            MaxPool2D((2, 2), padding='valid'),
            Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
            MaxPool2D((2, 2), padding='valid'),
            Conv2D(384, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            MaxPool2D((2, 2), padding='valid'),
            Conv2D(384, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            MaxPool2D((2, 2), padding='valid'),
            Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            Dropout(0.5),
        ]
        self.cnn = tf.keras.models.Sequential(layers)
        self.fc = tf.keras.layers.Dense(embedding_dim, activation='relu')

    def call(self, x):
        x = self.cnn(x)
        x = tf.reshape(x, (x.shape[0], -1, x.shape[3]))
        x = self.fc(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

# -----------------------------------------------------------------------------------------


def read_annotations(path: Path) -> list[tuple[str, str]]:
    rows = []
    with open(path, 'r') as f:
        for line in f.readlines():
            tmp = line.split()
            rows.append((tmp[0] + '.png', tmp[1]))
    return rows


def build_tokenizer(rows):
    """Same steps as the notebook, so ids map to the same letters."""
    words = ['<start> ' + ' '.join(list(word)) + ' <end>' for _, word in rows]
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(words)
    max_length = max(len(s) for s in tokenizer.texts_to_sequences(words))
    return tokenizer, max_length


def load_image(path: Path) -> tf.Tensor:
    img = tf.io.read_file(str(path))
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (HEIGHT, WIDTH))
    return img / 255.0


def make_decoder_fn(encoder, decoder, start_id: int, steps: int):
    @tf.function(input_signature=[tf.TensorSpec([1, HEIGHT, WIDTH, 3], tf.float32, name='image')])
    def decode(image):
        features = encoder(image, training=False)
        hidden = tf.zeros((1, decoder.units))
        dec_input = tf.constant([[start_id]], dtype=tf.int32)
        ids, probs = [], []
        for _ in range(steps):  # a Python loop, so the graph is unrolled: no while-loop to convert
            predictions, hidden, _ = decoder(dec_input, features, hidden, training=False)
            p = tf.nn.softmax(predictions[0])
            best = tf.argmax(p, output_type=tf.int32)
            ids.append(best)
            probs.append(tf.gather(p, best))
            dec_input = tf.reshape(best, (1, 1))
        return {'ids': tf.cast(tf.stack(ids), tf.int64), 'probs': tf.stack(probs)}

    return decode


def rename_outputs(model_proto):
    """tf2onnx names outputs after TF tensors; the server expects 'ids' and 'probs'."""
    import onnx

    renames = {}
    for out in model_proto.graph.output:
        elem = out.type.tensor_type.elem_type
        renames[out.name] = 'ids' if elem == onnx.TensorProto.INT64 else 'probs'
    for node in model_proto.graph.node:
        node.output[:] = [renames.get(n, n) for n in node.output]
        node.input[:] = [renames.get(n, n) for n in node.input]
    for out in model_proto.graph.output:
        out.name = renames[out.name]
    return model_proto


def ids_to_text(ids, tokens):
    out = []
    for i in ids:
        token = tokens[int(i)] if 0 <= int(i) < len(tokens) else ''
        if token == '<end>':
            break
        out.append(token)
    return ''.join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--annotations', type=Path, required=True, help='words_captcha/spec_train_val.txt')
    p.add_argument('--checkpoint', type=Path, help='checkpoint folder, e.g. checkpoints/train')
    p.add_argument('--out', type=Path, required=True, help='folder to write model.onnx and model.json')
    p.add_argument('--name', default='words-v1', help='model name shown by /v1/health')
    p.add_argument('--images', type=Path, help='image folder (default: next to the annotations)')
    p.add_argument('--eval', type=int, default=0, help='check accuracy on this many validation images')
    p.add_argument('--random-init', action='store_true', help='skip the checkpoint (only to test this script)')
    args = p.parse_args()

    if not args.random_init and not args.checkpoint:
        p.error('--checkpoint is required (or --random-init to test the script)')

    rows = read_annotations(args.annotations)
    image_dir = args.images or args.annotations.parent
    tokenizer, max_length = build_tokenizer(rows)
    vocab_size = len(tokenizer.word_index) + 1
    tokens = [tokenizer.index_word.get(i, '') for i in range(vocab_size)]
    print(f'{len(rows)} labelled images, vocab {vocab_size}, max length {max_length} tokens')

    encoder = CNN_Encoder(EMBEDDING_DIM)
    decoder = RNN_Decoder(EMBEDDING_DIM, UNITS, vocab_size)
    features = encoder(tf.zeros((1, HEIGHT, WIDTH, 3)))
    decoder(tf.zeros((1, 1), tf.int32), features, decoder.reset_state(1))

    if args.random_init:
        print('WARNING: random weights, the exported model will not read captchas')
    else:
        latest = tf.train.latest_checkpoint(str(args.checkpoint))
        if latest is None:
            raise SystemExit(f'No checkpoint found in {args.checkpoint}')
        status = tf.train.Checkpoint(encoder=encoder, decoder=decoder).restore(latest)
        status.expect_partial()  # the optimizer state in the checkpoint is not needed
        status.assert_existing_objects_matched()
        print(f'Restored {latest}')

    decode = make_decoder_fn(encoder, decoder, tokenizer.word_index['<start>'], max_length)

    import onnx
    import tf2onnx

    args.out.mkdir(parents=True, exist_ok=True)
    model_proto, _ = tf2onnx.convert.from_function(decode, input_signature=decode.input_signature, opset=15)
    model_proto = rename_outputs(model_proto)
    onnx.save(model_proto, str(args.out / 'model.onnx'))

    spec = {
        'name': args.name,
        'input': {'width': WIDTH, 'height': HEIGHT, 'channels': 3, 'layout': 'NHWC'},
        'decoder': 'ids',
        'tokens': tokens,
        'end_token': '<end>',
        'max_text_length': max_length - 2,
        'source': 'captcha_reader.ipynb attention model',
    }
    (args.out / 'model.json').write_text(json.dumps(spec, indent=2) + '\n')
    print(f'Wrote {args.out}/model.onnx and model.json')

    # Check the ONNX model gives the same answers as TensorFlow.
    import onnxruntime as ort

    session = ort.InferenceSession(str(args.out / 'model.onnx'), providers=['CPUExecutionProvider'])
    val = rows[NUM_TRAIN:] or rows
    sample = val[: max(args.eval, 20)]
    same = correct = 0
    for name, word in sample:
        path = image_dir / name
        if path.exists():
            x = load_image(path)[None]
        else:  # no images available (script test): use noise
            x = tf.random.uniform((1, HEIGHT, WIDTH, 3))
        tf_ids = decode(x)['ids'].numpy()
        onnx_ids = session.run(['ids'], {session.get_inputs()[0].name: x.numpy()})[0]
        same += int(np.array_equal(tf_ids, onnx_ids))
        correct += int(ids_to_text(onnx_ids, tokens) == word)
    print(f'ONNX matches TensorFlow on {same}/{len(sample)} images')
    if args.eval:
        print(f'ONNX exact match on {len(sample)} validation images: {correct / len(sample):.2%}')
    if same != len(sample):
        raise SystemExit('ONNX and TensorFlow disagree; do not deploy this export')


if __name__ == '__main__':
    main()
