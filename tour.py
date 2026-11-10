"""A narrated tour of what happens inside a small language model.

    uv run tour.py             # all nine parts
    uv run tour.py --part 3    # just one part
    uv run tour.py --part 7 8  # a few parts

Each part explains what you are about to see, prints real numbers from the
model, then says what to notice.

The NOTICE commentary was written for the default prompts below. If you change
them the numbers will still be right, but the commentary may no longer fit.
Finding out where it stops fitting is a good exercise; see README.md.
"""

import argparse
from textwrap import dedent
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "The capital of France is"
CONTRAST = "The capital of Spain is"  # same prompt with one token changed
DIMS = 8  # how many of the model's dimensions to print when showing raw numbers
BLOCK_TO_INSPECT = 10  # which block Part 5 takes apart (0-based)

parser = argparse.ArgumentParser(description="A narrated tour of what happens inside a small language model.")
parser.add_argument("--part", type=int, nargs="+", choices=range(1, 10), metavar="N", help="part(s) to show, 1-9 (default: all)")
args = parser.parse_args()

torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)

# =============================================================================
# Setup: load the model and record what happens inside it
# =============================================================================

tokenizer = AutoTokenizer.from_pretrained(NAME)
# float32 keeps the printed numbers clean. "eager" is the plain, slower
# attention code path; the fast default cannot hand back attention weights,
# which Part 6 needs.
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="eager")
model.eval()

# The four pieces of the model we care about. These attribute names are
# specific to Qwen (and other Llama-style models in Hugging Face). Another
# architecture has the same four pieces under different names: GPT-2, for
# example, calls them transformer.wte, transformer.h, transformer.ln_f, lm_head.
embed = model.model.embed_tokens  # lookup table: token id -> vector
blocks = model.model.layers  # the transformer blocks
final_norm = model.model.norm  # rescales the last vector before the readout
unembed = model.lm_head  # vector -> one score per vocabulary entry

# Hugging Face's `output_hidden_states` gives the stream after each block, but
# its last entry already has `final_norm` applied. To get every layer in the
# same raw form, and to see the attention and MLP updates separately, we
# attach hooks: small functions PyTorch calls whenever a module finishes.
captured = {}


def capture(key):
    def hook(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured[key] = tensor[0]  # [0] drops the batch dimension

    return hook


for i, block in enumerate(blocks):
    block.self_attn.register_forward_hook(capture(("attn", i)))
    block.mlp.register_forward_hook(capture(("mlp", i)))
    block.register_forward_hook(capture(("block", i)))


def run(text):
    """Run the model on `text` and return everything we recorded."""
    captured.clear()
    enc = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, output_attentions=True)
    n = len(blocks)
    ids = enc["input_ids"][0]
    return SimpleNamespace(
        ids=ids,
        tokens=[tokenizer.decode([t]) for t in ids],
        # stream[layer, position, dimension]; layer 0 = embeddings
        stream=torch.stack([out.hidden_states[0][0]] + [captured[("block", i)] for i in range(n)]),
        # the update each half of each block added: [block, position, dimension]
        attn=torch.stack([captured[("attn", i)] for i in range(n)]),
        mlp=torch.stack([captured[("mlp", i)] for i in range(n)]),
        # attention[block, head, from_position, to_position]
        attention=torch.stack([w[0] for w in out.attentions]),
        logits=out.logits[0],  # logits[position, vocabulary entry]
    )


def readout(vector):
    """Turn one residual-stream vector into a score for every vocab entry."""
    with torch.no_grad():
        return unembed(final_norm(vector))


def part(number, title, text):
    print(f"\n{'=' * 78}\nPART {number}: {title}\n{'=' * 78}")
    print(dedent(text).strip() + "\n")


def notice(text):
    print("\nNOTICE: " + dedent(text).strip().replace("\n", "\n        "))


def fmt(values, width=8):
    return " ".join(f"{v:{width}.3f}" for v in values.tolist())


# Run both prompts once. Every part below reads from these two recordings.
a = run(PROMPT)
c = run(CONTRAST)
assert len(c.ids) == len(a.ids), "CONTRAST must tokenize to the same length as PROMPT"

n_layers, n_pos, d_model = a.stream.shape[0] - 1, a.stream.shape[1], a.stream.shape[2]
n_heads = a.attention.shape[1]
last = n_pos - 1  # the final position: the one the prediction is read from
labels = [repr(t) for t in a.tokens]
norms = a.stream.norm(dim=-1)  # norms[layer, position]: length of each vector
subject = next(p for p in range(n_pos) if a.ids[p] != c.ids[p])  # where the prompts differ

# =============================================================================
# The tour
# =============================================================================


def part_1_tokens():
    part(1, "Text becomes tokens", f"""
        A model never sees letters. A tokenizer chops the text into pieces from
        a fixed vocabulary and hands the model the id number of each piece.

        Prompt: {PROMPT!r}
    """)
    for pos, (tok, tid) in enumerate(zip(labels, a.ids.tolist())):
        print(f"  position {pos}: {tok:<11} id {tid}")
    notice(f"""
        The space belongs to the token that follows it (' capital', not 'capital').
        From here on the model works with these {n_pos} positions, not with characters.
    """)


def part_2_embeddings():
    part(2, "Tokens become vectors (layer 0)", f"""
        The model owns a lookup table with one row per vocabulary entry:
        {embed.weight.shape[0]:,} rows x {d_model} numbers. Step one is to replace each
        token id with its row. That gives one vector of {d_model} numbers per
        position. This is "layer 0" of the residual stream.

        First {DIMS} of the {d_model} numbers for each token:
    """)
    for pos in range(n_pos):
        print(f"  {labels[pos]:<11} {fmt(a.stream[0, pos, :DIMS])}")
    same = torch.equal(a.stream[0], embed.weight[a.ids])
    print(f"\n  layer 0 is exactly the lookup-table rows for these ids: {same}")
    notice(f"""
        No context has been used yet. ' is' gets this same vector in every
        sentence ever written. Everything the model works out about THIS
        sentence has to be added on top by the {n_layers} blocks.
    """)


def part_3_grid():
    part(3, "The residual stream is a grid", f"""
        Picture a grid: one column per token, one row per layer. Every cell
        holds a vector of {d_model} numbers. Each block reads a row and produces
        the next one. That grid is the "residual stream":
        {n_layers + 1} rows x {n_pos} columns x {d_model} numbers.

        We can't print {d_model} numbers per cell, so here each cell is shrunk to
        one number: the vector's length (its "norm").
    """)
    print("  layer  " + "".join(f"{lab:>11}" for lab in labels))
    for layer in range(n_layers + 1):
        print(f"  {layer:5d}  " + "".join(f"{v:11.1f}" for v in norms[layer].tolist()))
    notice(f"""
        Every column starts tiny ({norms[0, last]:.1f}) and grows as blocks add to it.
        The first column is the odd one: it jumps to about {norms[:, 0].max():.0f} and sits
        there for most of the model. That is a known quirk of these models,
        not something about the word 'The'. The usual explanation: the model
        parks a giant, nearly constant vector on the first position, and
        attention rests on it when there is nothing useful to look at.
        Part 6 checks that. Ignore this column when comparing sizes.
    """)


def part_4_numbers():
    part(4, "Actual numbers in one column", f"""
        Now the real contents. This is the LAST column ({labels[last]}), because that
        is the vector the model will turn into its prediction. Each row is a
        layer; each column is one of the first {DIMS} dimensions (out of {d_model}).
        Read DOWN a column to follow one dimension through the model.
    """)
    print("  layer  " + " ".join(f"{'dim ' + str(d):>8}" for d in range(DIMS)))
    for layer in range(n_layers + 1):
        print(f"  {layer:5d}  {fmt(a.stream[layer, last, :DIMS])}")

    print("\n  Which dimensions are biggest, and how much of the vector do they hold?")
    for layer in (1, n_layers // 2, n_layers):
        v = a.stream[layer, last]
        top = v.abs().topk(3)
        share = (top.values**2).sum() / (v**2).sum()
        dims = ", ".join(f"dim {i} = {v[i]:.1f}" for i in top.indices.tolist())
        print(f"  layer {layer:2d}: {dims}   (top 3 hold {share:.0%} of the squared length)")
    notice(f"""
        1. Values drift from row to row. No row is a fresh rewrite of the one
           above it; Part 5 shows why.
        2. No column has a readable meaning. There is no "France dimension".
           Meaning lives in combinations of many dimensions (directions),
           which is why Parts 7 and 8 compare or decode whole vectors instead.
        3. The dimensions are very unequal. The {DIMS} shown here are small ones;
           just 3 of the {d_model} hold about a third of the vector, and dim 490 is
           in the top 3 at all three layers shown.
    """)


def part_5_blocks_add():
    b = BLOCK_TO_INSPECT
    before, after = a.stream[b, last], a.stream[b + 1, last]
    attn_update, mlp_update = a.attn[b, last], a.mlp[b, last]
    part(5, "Each block ADDS to the stream", f"""
        A block has two halves:
          attention - looks at earlier tokens and pulls information from them
          MLP       - computes on this position's vector alone
        Neither one replaces the vector. Each computes an update and adds it:

            after = before + attention_update + mlp_update

        That is why it is called a "stream": information accumulates in it.
        Check it on block {b + 1} (takes layer {b} to layer {b + 1}), last token, first {DIMS} dims:
    """)
    print(f"  before (layer {b:2d})   {fmt(before[:DIMS])}")
    print(f"  + attention update  {fmt(attn_update[:DIMS])}")
    print(f"  + MLP update        {fmt(mlp_update[:DIMS])}")
    print(f"  = sum               {fmt((before + attn_update + mlp_update)[:DIMS])}")
    print(f"  after  (layer {b + 1:2d})   {fmt(after[:DIMS])}")
    error = (a.stream[:-1] + a.attn + a.mlp - a.stream[1:]).abs().max().item()
    print(f"\n  largest mismatch over all {n_layers} blocks, all positions, all dims: {error:.6f}")

    print("\n  Size of each update at the last token:")
    print("  block   before   attention      MLP")
    for i in range(n_layers):
        print(f"  {i + 1:5d} {norms[i, last]:8.1f} {a.attn[i, last].norm():11.1f} {a.mlp[i, last].norm():8.1f}")
    mlp_bigger = int((a.mlp[:, last].norm(dim=-1) > a.attn[:, last].norm(dim=-1)).sum())
    notice(f"""
        1. "sum" and "after" match exactly, and the mismatch over the whole
           grid is zero. The stream really is just a running total of updates.
        2. After block 1, each update is smaller than the vector it is added
           to. A block nudges the stream; it does not replace it.
        3. At this position the MLP update is larger than the attention update
           in {mlp_bigger} of {n_layers} blocks.
        4. The last block adds an update of length {a.mlp[-1, last].norm():.1f} to a vector of
           length {norms[-2, last]:.1f}, and the result is SHORTER ({norms[-1, last]:.1f}). Adding can also
           cancel: the only way to remove something from the stream is to add
           its opposite.
    """)


def part_6_attention():
    weights = a.attention[:, :, last]  # [block, head, to_position] for the last token
    mean = weights.mean(dim=1)  # average over heads
    best_head = weights[:, :, subject].max(dim=1).values  # strongest head on the subject token
    part(6, "Where attention looks", f"""
        Attention is the only way information moves between columns. In each
        block, every position splits 100% of its attention across itself and
        the positions to its left, and pulls in information in that proportion.
        A block does this {n_heads} times in parallel ("heads"), each with its own
        pattern.

        Below: where the LAST token ({labels[last]}) looks in each block, averaged over
        the {n_heads} heads. The final column is the single head that looks hardest
        at {labels[subject]}.
    """)
    print("  block  " + "".join(f"{lab:>11}" for lab in labels) + f"   top head on {labels[subject]}")
    for i in range(n_layers):
        print(f"  {i + 1:5d}  " + "".join(f"{v:11.3f}" for v in mean[i].tolist()) + f"   {best_head[i]:10.2f}")
    first_wins = int((mean.argmax(dim=1) == 0).sum())
    strongest = ", ".join(f"block {i + 1} ({best_head[i]:.2f})" for i in best_head.topk(3).indices.sort().values.tolist())
    notice(f"""
        1. Each row adds up to 1: attention is a budget, not a set of
           independent dials.
        2. The first token takes the biggest share in {first_wins} of {n_layers} blocks,
           {mean[:, 0].mean():.0%} on average, even though 'The' says nothing about the
           question. This is the resting place from Part 3: a head with
           nothing to find parks there.
        3. Averages hide specialists. {labels[subject]} gets only {mean[:, subject].mean():.0%} on average, but
           single heads lock onto it: {strongest}.
           Heads like these are the candidates for carrying the country into
           this column. Proving which ones matter takes an experiment
           (switch a head off and see if the answer changes), not a table.
    """)


def part_7_contrast():
    part(7, "Changing one token: what moves and what cannot", f"""
        Compare our prompt with one that differs in a single token:
          A: {PROMPT!r}
          B: {CONTRAST!r}
        Both grids are compared cell by cell using cosine similarity:
          1.000 = the two vectors point the same way, 0 = unrelated.
        Column headers are prompt A's tokens; position {subject} is the one that differs.
    """)
    cos = torch.cosine_similarity(a.stream, c.stream, dim=-1)
    print("  layer  " + "".join(f"{lab:>11}" for lab in labels))
    for layer in range(n_layers + 1):
        print(f"  {layer:5d}  " + "".join(f"{v:11.3f}" for v in cos[layer].tolist()))
    left_diff = (a.stream[:, :subject] - c.stream[:, :subject]).abs().max().item()
    print(f"\n  largest difference anywhere in the columns left of the change: {left_diff:.6f}")
    notice(f"""
        1. Columns to the left of the changed token are identical at every
           layer. A position can only look at itself and what is to its left
           (Part 6), so nothing about the country can ever reach them.
        2. The changed column starts far apart ({cos[0, subject]:.2f}: two different words)
           and gets MORE alike going up. The model is building up what the
           two have in common: a country, in a sentence asking for a capital.
        3. The last column starts at exactly 1.000: same token, same lookup
           row. It only knows which country it is about because attention
           copied that in from the left. It stays close to 1 for most of the
           model, then drops in the final few layers ({cos[-1, last]:.2f} at the top).
           Compare that with where the answer shows up in Part 8.
    """)


def part_8_logit_lens():
    answer_a = a.logits[last].argmax().item()
    answer_c = c.logits[last].argmax().item()
    name_a, name_c = repr(tokenizer.decode([answer_a])), repr(tokenizer.decode([answer_c]))
    part(8, "Reading the stream before the model is finished", f"""
        Only the top row is ever turned into a prediction. But we can apply
        that same readout to lower rows and ask: "if the model stopped here,
        what would it say?" This trick is called the "logit lens".

        Caveat: the readout was trained to read the top row only. Low rows are
        not written in its "language", so the top token there is noise, not a
        thought.

        Columns: the top token at each layer for prompt A, then the
        probability and rank (out of {a.logits.shape[-1]:,}) of each prompt's final answer.
    """)
    print(f"  layer  {'top token (A)':<16} {'P(' + name_a + ')':>12} {'rank':>7}   {'P(' + name_c + ')':>12} {'rank':>7}")
    first_top = {}
    for layer in range(n_layers + 1):
        row = []
        for key, recording, answer in (("a", a, answer_a), ("c", c, answer_c)):
            scores = readout(recording.stream[layer, last])
            prob = scores.softmax(-1)[answer].item()
            rank = int((scores > scores[answer]).sum().item()) + 1
            if rank == 1:
                first_top.setdefault(key, layer)
            row.append((scores, prob, rank))
        top_token = repr(tokenizer.decode([row[0][0].argmax().item()]))
        print(f"  {layer:5d}  {top_token:<16} {row[0][1]:12.4f} {row[0][2]:7d}   {row[1][1]:12.4f} {row[1][2]:7d}")
    print(f"\n  {name_a} first becomes the top token at layer {first_top['a']}; {name_c} at layer {first_top['c']}.")
    notice("""
        1. Layer 0 reads back ' is': the embedding of a token decodes to
           itself. Then noise for the first half of the model, as the caveat
           said.
        2. The answer does not appear suddenly. Watch the Paris rank: it falls
           from the tens of thousands at layer 15 to 82 at layer 21, before
           taking first place at 22. The top token alone hides that build-up.
           Madrid follows the same shape a little later and only wins at the
           last layer.
        3. Most of layers 16-21 decode to blanks like ' ____'. Part 9 shows
           blanks are a real competitor in the final answer too: this model
           has read a lot of quiz worksheets ("The capital of France is ____").
           A reading, not a proven fact: the middle layers settle on "an
           answer goes here" before the last layers work out which answer.
        4. Both prompts are identical in this column at layer 0 and end up at
           different cities. That difference arrived through attention.
    """)


def part_9_generate():
    part(9, "From vector to word, then do it again", f"""
        The last step turns the top-right cell of the grid into a prediction:
          {d_model} numbers -> one score per vocabulary entry -> probabilities.
        To write more than one token, the model appends its choice to the
        prompt and runs the whole thing again. Two steps of that:
    """)
    text, previous = PROMPT, None
    for step in (1, 2):
        recording = run(text)
        top = recording.logits[-1].softmax(-1).topk(5)
        print(f"  step {step}: input = {text!r}")
        for p, t in zip(top.values.tolist(), top.indices.tolist()):
            print(f"      {p:7.1%}  {tokenizer.decode([t])!r}")
        if previous is not None:
            shared = previous.stream.shape[1]
            drift = (recording.stream[:, :shared] - previous.stream).abs().max().item()
            print(f"\n  largest change in the first {shared} columns after adding a token: {drift:.6f}")
        previous = recording
        text += tokenizer.decode([top.indices[0].item()])
    print(f"\n  result: {text!r}")
    notice(f"""
        1. The output is a probability for every vocabulary entry, not one
           answer. Here we always took the top one; chat apps usually sample
           from the list.
        2. This is a "base" model: it continues text, it does not answer
           questions. That is why quiz-style blanks get a real share.
        3. Adding a token left the first {n_pos} columns unchanged (the tiny number
           is rounding noise). Same rule as Part 7: nothing to the right can
           affect them. Real systems use this to avoid recomputing old columns.
    """)


PARTS = [
    part_1_tokens,
    part_2_embeddings,
    part_3_grid,
    part_4_numbers,
    part_5_blocks_add,
    part_6_attention,
    part_7_contrast,
    part_8_logit_lens,
    part_9_generate,
]

print(f"Model: {NAME}  ({n_layers} blocks, {n_heads} heads per block, {d_model} dimensions)")
for number in args.part or range(1, len(PARTS) + 1):
    PARTS[number - 1]()
if not args.part:
    print("\nThat's the tour. README.md has things to try next and where to go deeper.")
