# Reducing model risk

## The risk

A captcha model learns one style: its fonts, distortion, noise and characters. The
notebook model was trained on `words_captcha` only, so it can do 90.7% on that style and
close to 0% on a different one. Each customer site has its own captcha, so every new
customer can be a style the model has never seen. If you promise "we solve your captcha"
before you've measured it on their captcha, you will sell something that doesn't work.

Below, in order: measure first, train per style, watch it in production, and limit what
you promise.

## 1. Measure before you take a customer

For every new customer, before they pay:

1. Collect 300 to 1,000 of their captcha images and the correct answers. Ask the customer,
   or save images from their site and label them by hand. Keep them in the same format as
   the training data: a folder of PNGs plus `labels.txt` with `<name> <text>` lines.
2. Split them: about 80% for training, 20% kept back as a test set you never train on.
3. Run the current model on the test set:

   ```bash
   python -m training.evaluate --model models/current --data datasets/acme-test/labels.txt --failures acme-failures.csv
   ```

   It prints exact-match accuracy and a table of confidence cut-offs (see step 4).
4. Agree on a number with yourself before you look at it, for example "at least 85% right
   on the captchas we answer, answering at least 80% of them". Below that, train first
   (section 2) or turn the customer down.

`evaluate.py` runs the same `Solver` code as the API, so its numbers are what customers
will get.

## 2. Train for the style, starting from little data

Use the CRNN + CTC trainer in `training/train_ctc.py`. It reads the whole word in one pass,
handles any characters (digits, upper case, symbols) and any length, and exports straight
to the folder layout the server loads.

- **Have a few hundred real samples?** Train on them plus synthetic padding, and validate
  on real ones only:

  ```bash
  python -m training.train_ctc \
      --data datasets/acme-train/labels.txt datasets/synth-train/labels.txt \
      --val datasets/acme-test/labels.txt --name acme-v1 --out runs/acme-v1
  ```

- **Make the synthetic data look like their captcha.** `training/synth.py` uses the
  `captcha` library's default style. Change its fonts, colours, noise and size to resemble
  the customer's images. The closer the synthetic data looks, the fewer real samples you
  need.
- **Mind the characters.** If their captcha has digits or capitals, the labels must too,
  and the docs must stop saying "lowercase a–z" for that customer.
- **Upper/lower case.** Many sites check answers without caring about case. Ask. If they
  don't care, train with `--lowercase`: it is easier for the model and more accurate.

One model per style is normal, and the server supports it: put each model in its own folder
under `models/` and give the customer's key that model with
`c2t-keys set-model <id> acme-v1`. Keys without a model use `models/current`.

## 2b. A general default for everyone else

Until a customer has their own model, their key uses the default model. That is now
ddddocr's pretrained general model (`models/ddddocr-beta-v1`, MIT licence, imported with
`training/import_ddddocr.py`), limited to letters and lowercased to match the API docs.

On our test sets it read 95.1% of a "simple" synthetic style that no model of ours had
seen, where our CRNN trained on another style got 0%. On the heavily warped
captcha-library style it got 47.5%, where our CRNN trained on that style got 98.4%. So:
ddddocr is a reasonable answer on day one, and a model trained on the customer's own
captchas is what gets you to 95%+ on hard styles. Measure both on the customer's samples
(section 1) and assign whichever wins.

## 3. Let customers skip answers that are probably wrong

Every answer has a `confidence`. `evaluate.py` shows, for each cut-off, how many captchas
you would answer and how many of those would be right. Example of reading it:

```
 cut-off  answered  right when answered
    0.00    100.0%               88.0%
    0.80     86.0%               97.5%
```

Tell each customer their cut-off: below it, they reload the captcha and try again instead
of submitting a likely-wrong answer. A second try costs them one request; a wrong submit
may lock their account on the target site.

## 4. Watch it in production

- `c2t-keys usage` shows solved and failed requests per key. A sudden rise in failures, or
  a drop in average confidence, usually means the site changed its captcha.
- What the API can't see is whether the site accepted the answer. Add a
  `POST /v1/report` endpoint (with the `request_id`) that customers call when an answer was
  rejected. That gives you a real accuracy number per customer, and, if your terms allow
  you to keep the images, new training data with the hardest cases in it.
- Re-run `evaluate.py` on each customer's test set every time you retrain, and before you
  switch `models/current`. A new model must not be worse on any customer's test set.

## 5. Release models safely

- Keep each model in its own folder (`models/words-v1`, `models/acme-v1`) and point
  `models/current` at the live one. Rolling back is pointing it back and restarting.
- Never delete the previous model until the new one has run for a week.
- `model.json` records the training data, sample counts and validation accuracy, so you
  can always tell what is running (`GET /v1/health` shows the name).

## 6. Limit what you promise

- The docs already say what is supported: text captchas only, PNG/JPEG, and which
  characters. reCAPTCHA, hCaptcha, Turnstile and image puzzles are out of scope; don't
  sell them.
- Quote accuracy per customer from their own test set, never the 90.7% from
  `words_captcha`.
- Keep the acceptable-use terms and revoke keys that break them. Solving captchas on sites
  the customer doesn't own or have permission for can break those sites' terms, and that
  risk lands on you as the provider (hosting, payment processors, reputation). Knowing who
  your customers are and what sites they use the API on is part of reducing risk.
