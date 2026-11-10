# residual-stream-tour

A notebook that runs a small language model and narrates what happens inside
it, from the text you type to the word it predicts.

**[Read the tour → `tour.ipynb`](tour.ipynb)**

GitHub shows the notebook with all of its output and charts, so there is
nothing to install: open it and read from top to bottom.

This is a first look, not a course or a library: one prompt, one small model,
plain PyTorch and Hugging Face `transformers`. For real experiments, see
[Where to go next](#where-to-go-next).

## What's in it

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

Each part explains what you are about to see, shows real numbers or a chart
from inside the model, then says what to notice. The notebook ends with a
list of things to try.

## Run it yourself

You need [uv](https://docs.astral.sh/uv/). Clone this repo, then:

```
uv run --with jupyter jupyter lab tour.ipynb
```

In Jupyter, choose **Run → Run All Cells**. The settings (the prompts and the
model) are in the first code cell.

The first run downloads the model,
[Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) (about 1 GB), into
`~/.cache/huggingface`. No GPU or account is needed, and the whole notebook
runs in well under a minute.

Other setups should work but have not been tested: the notebook only needs
`torch`, `transformers` and `matplotlib`, so plain `pip` with any Jupyter, or
[Google Colab](https://colab.research.google.com/github/ciminelli/residual-stream-tour/blob/main/tour.ipynb),
are worth a try.

## Caveats

- Everything here is an observation about one prompt in one small model.
- The charts and tables show what is in the model, not what causes its
  answer. Where the commentary interprets rather than measures, it says so.
- The commentary was written for the default prompts. Change them and the
  numbers stay right, but the commentary may not.
- The logit lens decodes middle layers with a readout trained only on the
  last one, so early layers decode to noise.
- The attribute names in the notebook (`model.model.layers` and so on) are
  specific to Qwen and other Llama-style models in Hugging Face. The setup
  section gives GPT-2's equivalents.

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
