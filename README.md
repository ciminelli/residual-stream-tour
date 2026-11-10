# residual-stream-tour

One Python file that runs a small language model on your computer and
narrates what happens inside it, from the text you type to the word it
predicts.

```
uv run tour.py
```

It prints nine short parts. Each one says what you are about to see, shows
real numbers from inside the model, then says what to notice.

This is a first look, not a course or a library: one prompt, one small model,
plain PyTorch and Hugging Face `transformers`. Run it, read the output, then
change things. For real experiments, see [Where to go next](#where-to-go-next).

## Run it

You need [uv](https://docs.astral.sh/uv/). Clone this repo, then:

```
uv run tour.py             # all nine parts
uv run tour.py --part 3    # just one part
uv run tour.py --part 7 8  # a few parts
uv run tour.py | less      # one screen at a time
```

The first run downloads the model,
[Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) (about 1 GB), into
`~/.cache/huggingface`. No GPU or account is needed, and later runs take a
few seconds.

No uv? `pip install torch transformers` then `python tour.py` should work,
but only the uv route on Python 3.13 has been tested.

## The tour

The parts follow the path a prompt takes through the model.

| Part | What it shows | The idea |
|---|---|---|
| 1 | Text → tokens | The model works on token ids, not characters. |
| 2 | Tokens → vectors | Each token starts as a row from a lookup table, with no context. |
| 3 | The grid | One vector per token per layer. That grid is the "residual stream". |
| 4 | Raw numbers | What the vectors contain, and why you can't read single dimensions. |
| 5 | Blocks add | `after = before + attention update + MLP update`, checked on real numbers. |
| 6 | Attention | Where the last token looks, and why averages hide the interesting heads. |
| 7 | One token changed | France vs Spain: which cells change, and which cannot. |
| 8 | Logit lens | Decoding every layer to watch the answer form. |
| 9 | Prediction | Vector → probabilities → next token, then repeat to generate text. |

A sample from Part 8. The prompt is `The capital of France is`, and each row
asks "if the model stopped at this layer, how likely is `' Paris'`?":

```
  layer  top token (A)     P(' Paris')    rank
     15  '\';";\n'              0.0000   33348
     16  ' ____'                0.0000    6412
     17  ' ____'                0.0000    9761
     18  ' ____'                0.0000    3552
     19  '?>>'                  0.0001    1358
     20  ' __'                  0.0002     488
     21  ' __'                  0.0004      82
     22  ' Paris'               0.2493       1
     23  ' Paris'               0.3439       1
     24  ' Paris'               0.3156       1
```

## Things to try

The settings are constants at the top of `tour.py`. The `NOTICE` commentary
was written for the defaults: change something and the numbers stay right,
but the commentary may not. Spotting where it stops being true is the
exercise. If you change `PROMPT`, change `CONTRAST` to match; the two must
have the same number of tokens.

1. **Change the contrast country.** Set `CONTRAST = "The capital of Japan is"`.
   The model's top guess is no longer a city. What is it, and what does
   Part 8 look like now?
2. **Get rid of the blanks.** The model gives real probability to tokens like
   `' ____'`. Give it an example first,
   `"The capital of Italy is Rome. The capital of France is"`, and compare
   Part 9.
3. **Take apart a different block.** Change `BLOCK_TO_INSPECT` to `0` or `23`
   and compare the updates in Part 5.
4. **Find a prompt it gets wrong.** Does the right answer still show up in
   the middle layers of Part 8?
5. **Harder: test a claim.** Part 6 names attention heads that look hard at
   `' France'`. Switch one off and see whether the prediction changes.

## Caveats

- Everything here is an observation about one prompt in one small model.
- The tables show what is in the model, not what causes its answer. Where the
  commentary interprets rather than measures, it says so.
- The logit lens decodes middle layers with a readout trained only on the
  last one, so early layers decode to noise.
- The attribute names in `tour.py` (`model.model.layers` and so on) are
  specific to Qwen and other Llama-style models in Hugging Face. A comment in
  the file gives GPT-2's equivalents.

## Where to go next

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html):
  the paper that lays out the residual-stream view used here.
- [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens):
  the original write-up of the trick in Part 8.
- [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762)
  and [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453):
  the giant first-token vector in Part 3 and the attention that rests on it
  in Part 6.
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens):
  a library for this kind of work, including switching parts of a model off.
- [ARENA](https://www.arena.education/): free course materials with a
  hands-on interpretability chapter.

## License

The code is MIT licensed; see [LICENSE](LICENSE). The model is downloaded
separately from Hugging Face under its own license (Apache 2.0).
