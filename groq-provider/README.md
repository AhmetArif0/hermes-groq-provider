# Groq provider for Hermes

Use [Groq](https://groq.com)'s models in [Hermes Agent](https://github.com/NousResearch/hermes-agent) as a
first-class provider: pick **Groq** in `hermes model`, or run
`hermes chat --provider groq -m openai/gpt-oss-120b`.

An independent community plugin, not affiliated with Groq.

## Why not a custom endpoint?

Hermes can already point a custom OpenAI-compatible endpoint at Groq. Measured on 2026-09-29 against
Groq's live API with Hermes v0.21.5 and `main`:

| | Custom endpoint | This plugin |
|---|---|---|
| `openai/gpt-oss-120b`, `openai/gpt-oss-20b` | Every turn fails, whatever the reasoning setting: Hermes sends `reasoning_effort: "default"` (`"none"` when reasoning is off) and Groq answers HTTP 400 (`reasoning_effort must be one of low, medium, or high`) | Works, and your reasoning setting is sent in the form the model accepts |
| Model list | All 11 models Groq returns, including speech-to-text, text-to-speech, prompt-guard and 4K-context models that Hermes cannot run | The 4 models Hermes can run |

An open Hermes pull request, [#124103](https://github.com/NousResearch/hermes-agent/pull/124103), would
send gpt-oss-120b and gpt-oss-20b a graded level on custom endpoints. The plugin does not depend on it and
works the same with or without it.

## Install

```sh
hermes plugins install groq-provider
```

Get an API key at [console.groq.com/keys](https://console.groq.com/keys) and add it to `~/.hermes/.env`:

```sh
GROQ_API_KEY=gsk_...
```

Then choose **Groq** in `hermes model`, or set it in `~/.hermes/config.yaml`:

```yaml
model:
  provider: groq
  default: openai/gpt-oss-120b
```

## Models and reasoning effort

The model list comes from Groq's API, minus the models Hermes cannot run: whisper (speech-to-text),
orpheus (text-to-speech), prompt-guard (a 512-token classifier) and allam-2-7b (a 4,096-token window;
Hermes needs at least 64K). On 2026-09-29 that left:

| Model | Reasoning effort Hermes sends |
|---|---|
| `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b` | `low`, `medium` or `high`. These models always reason, so `none` (off) and `minimal` become `low`; `xhigh`, `max` and `ultra` become `high` |
| `qwen/qwen3.8-27b` | `none`, `low`, `medium` or `high`. `minimal` becomes `low`; anything above `high` becomes `high` |
| Any other model | Nothing: Groq rejects the field on models without reasoning |

With no reasoning effort configured, nothing is sent and Groq uses the model's default. A level is only
ever lowered to the nearest one the model accepts, never raised.

## Groq's free plan

Groq's free plan allows 8,000 tokens per minute per model (6,000 for allam-2-7b), as reported in its
rate-limit headers on 2026-09-29. One Hermes request with tools is larger than that: about 18.8K
tokens with the default toolsets, and 10.7K to 11.7K with a single toolset. On the free plan Groq therefore
rejects each agent turn with HTTP 413 (`Request too large ... on tokens per minute (TPM)`), which Hermes
reports as the conversation having grown too large to send. A turn without tools fits. For agent
work you need Groq's paid Developer plan. This is Groq's limit; the plugin does not change the size of
a request.

## Security and footprint

- Registers one model provider, `groq`, when Hermes loads it. No tools, hooks, commands or background
  threads.
- Network: chat requests and the model list go to `https://api.groq.com/openai/v1` (or the
  `model.base_url` you set) through Hermes' own client, carrying `GROQ_API_KEY`. The plugin itself
  opens no connections.
- Reads no files, writes no files, starts no processes and has no dependencies beyond Hermes. Hermes,
  not the plugin, reads `GROQ_API_KEY`.

## Compatibility

Hermes 0.21.5 or newer. CI runs the tests against Hermes v2026.9.24 (0.21.5) and `main` every day.

## Changelog

### 1.0.0

- The `groq` model provider: per-model reasoning effort, and a model list without the models Hermes
  cannot run.

## License

[MIT](https://github.com/AhmetArif0/hermes-groq-provider/blob/main/LICENSE)
