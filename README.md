### GPT Slack Agent scaffolding

An opinionated Go scaffolding to build Slack agents powered by LLMs. It wires together:

- **Slack Socket Mode** event ingestion and posting utilities
- **LLM helper** supporting both **OpenAI Chat Completions** and the **Anthropic Messages API**
- **MCP** connection helper
- Optional **Gmail** polling utilities for email-driven workflows

You bring a small `main.go` that plugs in your business logic (Slack and/or Mail processors). This package provides the plumbing so you can focus on your agent behavior.

### Features

- **Socket Mode Slack client** with helpers to post to channels/threads, fetch thread replies, post/remove reactions and basic text formatting
- **Event filter** that forwards `app_mention` and plain `message` events for your processing
- **LLM client** wrapper with a simple `GptQuery` API and sensible defaults, backed by either OpenAI or Anthropic
- **Provider-neutral stop reasons** via `Query`, so continuation loops work the same on either provider
- **Tool use** behind the same interface: hand the model a set of tool definitions, get back the calls it wants to make, and run the loop with `RunToolLoop` — MCP servers plug straight in
- **Gmail** utilities for polling labeled messages and parsing bodies (plain and HTML)
- **MCP client** with support for Streamable, SSE, and STDIO transports for Model Context Protocol integration
- **Notion MCP** integration with specialized client for Notion's MCP implementation
- **Embedding utilities** with support for local ONNX models and OpenAI embeddings for RAG systems
- **YAML config** loader, including pass-through `agent_config` for your custom settings

### Repository layout

- `agent/` — Core agent wiring: config loader, Slack client initialization, email loop, LLM factory, MCP integration
  - `mcp.go` — Model Context Protocol client implementation with multiple transport options, and the bridge that turns MCP tools into ones the LLM can call
  - `notionmcp.go` — Specialized Notion MCP client implementation
  - `headers.go` — HTTP header utilities for MCP clients
- `slack/` — Slack client and helpers (`PostInChannel`, `PostInThread`, `GetThreadMessages`, `StripAtMention`, `AddText`)
- `gpt/` — LLM helpers behind a common `LLM` interface (`GptQuery`, `Query`, `NewChat`)
  - `llm.go` — The `LLM` interface, `Response` type, stop reasons, and the provider factory
  - `tools.go` — Provider-neutral tool definitions, the `Chat` interface and `RunToolLoop`
  - `gpt.go` — OpenAI Chat Completions, plus embeddings (`GetEmbedding`, `GetEmbeddingsBatch`)
  - `claude.go` — Anthropic Messages API
- `embedding/` — Embedding generation and RAG utilities (local ONNX models and OpenAI embeddings)
- `mail/` — Gmail connection and parsing utils
- `config.yaml` — Example configuration

### Requirements

- Go 1.20+
- A Slack app with Bot Token and App-Level Token (Socket Mode)
- An OpenAI API key or an Anthropic API key (an OpenAI key is additionally required for embeddings)
- Optional Gmail OAuth credentials JSON (if using email polling)

### Install and setup

1) Initialize a Go module for your app and add this code as a dependency or local module.

```bash
mkdir my-slack-agent && cd my-slack-agent
go mod init example.com/my-slack-agent
# If using locally, replace the module path to match your fork or add a replace directive.
```

If you fork/rename this module, make sure your imports match your module path (examples below use `github.com/vtuson/slackagent/...`). You can either keep that module path or update imports to your own module path.

2) Copy `config.yaml` and fill in secrets.

```bash
cp /path/to/this/repo/config.yaml ./config.yaml
```

3) Create your `main.go` and wire the processors.

```go
package main

import (
    "log"
    "strings"

    agt "github.com/vtuson/slackagent/agent"
    agtmail "github.com/vtuson/slackagent/mail"
    "github.com/vtuson/slackagent/slack"
    "github.com/slack-go/slack/slackevents"
)

func main() {
    a := &agt.Agent{}
    if err := a.LoadConfig("config.yaml"); err != nil { log.Fatal(err) }

    // Initialize Slack (Socket Mode) and set the event processor
    a.SlackProcessor = func(evt interface{}) {
        client := a.GetSlackClient()
        switch ev := evt.(type) {
        case *slackevents.AppMentionEvent:
            // Clean message and query LLM
            prompt := slack.StripAtMention(ev.Text)
            llm := a.NewLLM()
            reply, err := llm.GptQuery("You are a helpful Slack bot.", prompt, "")
            if err != nil { reply = "Sorry, I had an issue answering that." }
            // Post reply in thread
            if strings.TrimSpace(ev.ThreadTimeStamp) == "" {
                ts, _ := client.PostInChannel(ev.Channel, reply)
                _ = ts
            } else {
                _, _ = client.PostInThread(ev.Channel, reply, ev.ThreadTimeStamp)
            }

        case *slackevents.MessageEvent:
            // Optionally react to plain messages
            // ... your logic ...
        }
    }

    a.InitializeSlackClient()

    // Optional: email loop if configured
    a.EmailProcessor = func(email agtmail.Email) {
        // Handle incoming email -> Slack or LLM
        // ... your logic ...
    }
    if a.HasEmail() {
        go a.ProcessEmails()
    }

    // Wait for Ctrl+C
    a.WaitForSignal()
}
```

Note: If you change the module path, update the imports accordingly. The example uses the module path declared in this repo's imports.

### Configuration

Use a YAML file like `config.yaml` (an example exists at the repo root):

```yaml
mail:
  label: "mediate"        # Gmail label to read
  maxid: "NO_ID"          # Start from latest; set to a concrete message ID to resume
  wait: 15                 # Poll interval in minutes
  secret: "client_secret.json"  # Path to Gmail OAuth client secret JSON
  auth_token: ""          # Optional: one-time authorization code for first run

slack:
  token: "xoxb-..."       # Bot token
  app_token: "xapp-..."   # App-level token (Socket Mode)
  channel: "CXXXXXXX"     # Default channel to post

gpt:
  key: "sk-..."           # OpenAI or Anthropic API key
  model: "gpt-3.5-turbo"  # Model name
  provider: ""            # Optional: "openai" or "anthropic". Inferred from the model when empty
  max_tokens: 0           # Optional: reply cap. 0 leaves the provider default
  temperature: 0.7        # Optional: sampling temperature. Omit to leave unset
  effort: "medium"        # Optional: reasoning depth. Omit to leave unset

mcp:
  notion:
    key: "secret_..."     # Notion API Key
    impl_name: "my-app"   # Implementation name for MCP client
    impl_version: "v1.0"  # Implementation version
    url: ""              # Optional: custom MCP endpoint URL

agent_config:              # Free-form config for your app
  my_setting: 123
  feature_flag: true
```

Access your custom `agent_config` via:

```go
type MyConfig struct { MySetting int `yaml:"my_setting"`; FeatureFlag bool `yaml:"feature_flag"` }
var cfg MyConfig
_ = a.GetCustomConfig(&cfg)
```

### Choosing an LLM provider

`a.NewLLM()` returns a `gpt.LLM`, backed by either OpenAI or Anthropic. Which one you get is decided by the `gpt` block in your config.

**By model name (no provider needed).** Any model starting with `claude-` is routed to Anthropic, anything else to OpenAI. Configs written before provider support existed keep working unchanged:

```yaml
gpt:
  key: "sk-ant-..."
  model: "claude-opus-5"    # inferred as anthropic
```

**Explicitly**, when you want to be unambiguous:

```yaml
gpt:
  key: "..."
  model: "..."
  provider: "anthropic"     # or "openai"
```

`max_tokens` is optional and caps the reply length. Left at `0` (or omitted), each provider keeps its own default — `1024` for Anthropic, which requires the field, and omitted entirely for OpenAI, which does not:

```yaml
gpt:
  max_tokens: 4096
```

An unknown provider name fails at `LoadConfig` time rather than on the first query.

### Tuning knobs: temperature and effort

Both providers removed the sampling parameters from their newest models and
replaced them with a reasoning-effort dial, so the two knobs are very nearly
mutually exclusive:

|                          | `temperature` | `effort` |
| ------------------------ | ------------- | -------- |
| `gpt-3.5` / `gpt-4o`     | yes           | no       |
| o-series / `gpt-5`       | no            | yes      |
| `claude-3.x` / `haiku-4-5` | yes         | no       |
| `claude-opus-5`          | no            | yes      |

There is no knob that works across the whole matrix, so setting one the chosen
model rejects would be a 400 on the first Slack mention. Instead it fails at
`LoadConfig` with a message naming the model:

```
Invalid llm configuration: model "claude-opus-5" does not accept temperature;
it was removed on this model in favour of effort, so drop the temperature
setting or pick an older model
```

`temperature` is passed through **provider-native and is not rescaled**, so the
same number means different things: OpenAI's range is 0–2 with a default of 1,
Anthropic's is 0–1. A value copied from one config to the other will not behave
the same.

Omitting `temperature` leaves it unset, which is deliberately different from
setting it to `0` — zero is a real value asking for the most deterministic
output. In Go this is why the config field is a `*float64`:

```yaml
gpt:
  model: "gpt-4o"
  temperature: 0        # sent as temperature=0
```

`effort` is one of `low`, `medium`, `high`, `xhigh`, `max`. OpenAI stops at
`high`; the top two are Anthropic-only. It maps onto `output_config.effort` for
Anthropic and `reasoning_effort` for OpenAI:

```yaml
gpt:
  model: "claude-opus-5"
  effort: "low"         # cheaper and faster on routine mentions
```

Which models accept which knob is a pair of prefix tables — `claudeNoSampling`
and `claudeEffort` in `gpt/claude.go`, `openaiReasoning` in `gpt/gpt.go`. They
go stale whenever a provider ships a model, and they are the only place to edit
when that happens.

To check from code rather than config, ask the provider:

```go
llm := a.NewLLM()
if llm.Supports(gpt.KNOBEFFORT) {
    llm.SetEffort(gpt.EFFORTLOW)
}
```

#### Asking a question

`GptQuery` returns the reply text, and treats a refusal or an empty reply as an error:

```go
llm := a.NewLLM()
reply, err := llm.GptQuery("You are a helpful Slack bot.", prompt, "")
```

The third argument is optional extra context. OpenAI receives it as an additional user message; Anthropic receives it appended to the user turn.

#### Stop reasons and continuation loops

`Query` returns the reply together with why generation stopped, which is what you need to continue a reply that ran out of room:

```go
var full string
for {
    resp, err := llm.Query(system, prompt, full)
    if err != nil {
        return err
    }
    full += resp.Text
    if !resp.Truncated() {
        break
    }
}
```

`Response` carries:

| Field | Meaning |
|---|---|
| `Text` | The reply text |
| `StopReason` | Normalised — one of the `STOP*` constants below |
| `RawStopReason` | The provider's own value, for logging |
| `Detail` | Provider explanation, e.g. why a request was refused. Usually empty |

Providers spell stop reasons differently, so they are normalised onto a shared set and a loop written against one provider behaves the same on the other:

| Constant | OpenAI | Anthropic |
|---|---|---|
| `gpt.STOPEND` | `stop` | `end_turn`, `stop_sequence` |
| `gpt.STOPMAXTOKENS` | `length` | `max_tokens` |
| `gpt.STOPTOOLUSE` | `tool_calls`, `function_call` | `tool_use` |
| `gpt.STOPREFUSAL` | `content_filter` | `refusal` |
| `gpt.STOPOTHER` | anything else | anything else |

`resp.Truncated()` is shorthand for `StopReason == gpt.STOPMAXTOKENS`.

`Query` and `GptQuery` differ in how they treat a refusal: `Query` reports it as data, so a loop can branch on it, while `GptQuery` returns an error.

> **Note:** `Query` is single-turn — it does not carry message history. The loop above feeds the text so far back through the `context` argument, which is an approximation of a real continuation rather than resuming an assistant turn. It is enough to detect and react to truncation. For true multi-turn work, use `NewChat` below, which keeps the conversation.

#### Tool use

`gpt.STOPTOOLUSE` tells you the model wants to call a tool, but a tool call cannot be answered within a single request: the model asks, and the answer only reaches it on a second request that still carries the turn it asked in. That is what `NewChat` is for — `Query` and `GptQuery` are single-turn and cannot offer tools at all.

```go
chat := llm.NewChat("You are a helpful Slack bot.", []gpt.Tool{{
    Name:        "get_page",
    Description: "Fetch a page by name",
    InputSchema: map[string]any{
        "type":       "object",
        "properties": map[string]any{"page": map[string]any{"type": "string"}},
        "required":   []any{"page"},
    },
}})

resp, err := gpt.RunToolLoop(chat, prompt, func(call gpt.ToolCall) gpt.ToolResult {
    args, err := call.Arguments()
    if err != nil {
        return gpt.ToolResult{ID: call.ID, Content: err.Error(), IsError: true}
    }
    return gpt.ToolResult{ID: call.ID, Content: fetchPage(args["page"].(string))}
}, 0)
```

`RunToolLoop` sends the message, runs whatever the model asks for, feeds the results back, and repeats until the model answers. Its last argument caps the number of tool rounds, so a model that keeps asking cannot bill forever; `0` means `gpt.DEFAULTTOOLTURNS`.

To approve or inspect calls first, drive the `Chat` directly:

```go
resp, err := chat.Send(prompt)
for resp.WantsTool() {
    var results []gpt.ToolResult
    for _, call := range resp.ToolCalls {
        // decide whether to run it, then:
        results = append(results, gpt.ToolResult{ID: call.ID, Content: output})
    }
    resp, err = chat.SendToolResults(results)
}
```

Two rules the providers both enforce: every call in a reply must be answered, and they must all be answered in the same turn. Report a failed tool as `ToolResult{IsError: true}` rather than aborting — the model can read the message and try something else, which it cannot do if the conversation stops.

| Type | Meaning |
|---|---|
| `gpt.Tool` | A tool offered to the model: name, description, and a JSON Schema for the arguments |
| `gpt.ToolCall` | The model asking: `ID`, `Name`, and raw JSON `Input` (decode with `Arguments()`) |
| `gpt.ToolResult` | What the tool produced, keyed by the call's `ID`, with `IsError` for failures |
| `gpt.Chat` | The conversation: `Send` and `SendToolResults` |

The wire formats differ and the differences are handled for you: Anthropic takes the schema at the top level and has an `is_error` flag, while OpenAI nests it under `function.parameters`, encodes arguments as a JSON string, and has no error flag (failures are prefixed into the content instead). Each provider keeps its own history in its native format, so an assistant turn goes back exactly as it arrived. A `Chat` is not safe for concurrent use — give each conversation its own.

#### Tools from an MCP server

An MCP server already publishes tool definitions with JSON Schemas, so no conversion is needed on your side:

```go
tools, err := a.MCPClient.Tools(ctx)
if err != nil {
    return err
}
chat := llm.NewChat(systemPrompt, tools)
resp, err := gpt.RunToolLoop(chat, prompt, a.MCPClient.ToolRunner(ctx), 0)
```

`ToolRunner` dispatches each call to the server and packages the outcome, marking failures as errors for the model instead of returning them to you. Use `RunTool` if you want to dispatch a single call yourself.

#### Embeddings are OpenAI-only

Anthropic has no embeddings endpoint, so embeddings stay on the OpenAI client regardless of which provider handles chat. Use `a.NewEmbedder()`, which returns an error rather than silently producing garbage when the agent is configured for Anthropic:

```go
embedder, err := a.NewEmbedder()
if err != nil {
    // configured for Anthropic — supply an OpenAI key if you need embeddings
}
vec, err := embedder.GetEmbedding("some text")
```

If you need Claude for chat *and* embeddings, construct a `gpt.OpenAI` directly with an OpenAI key alongside your Anthropic config.

### Slack bot configuration (api.slack.com)

- Create a new app in `api.slack.com/apps`
- Create an **App-Level Token**, add scope `connections:write` and save the token (`xapp-...`)
- Go to **Socket Mode** and enable it
- Go to **OAuth & Permissions** and add the following OAuth scopes, then install the app to your workspace to obtain the **Bot Token** (`xoxb-...`)

| OAuth Scope            | Description |
|------------------------|-------------|
| `app_mentions:read`    | View messages that directly mention the app in conversations the app is in |
| `channels:history`     | View messages and other content in public channels the app has been added to |
| `channels:read`        | View basic information about public channels in a workspace |
| `chat:write`           | Send messages as the app |
| `chat:write.customize` | Send messages as the app with a customized username and avatar |
| `reactions:read`       | View emoji reactions and their associated content in channels and conversations the app has been added to |
| `incoming-webhook`     | Post messages to specific channels in Slack |

- Under **Event Subscriptions**, enable and subscribe to events you need (for this agent, at least `app_mention`; you may also use `message.channels`)
- Put your default channel ID under `slack.channel` in `config.yaml`

### Gmail setup (optional)

- Create OAuth 2.0 credentials in Google Cloud Console and download the client JSON
- Save it (e.g. `client_secret.json`), and configure `mail.secret`
- First run without `mail.auth_token` prints an auth URL and exits; complete the flow and re-run with the code in `mail.auth_token` to cache the token. Subsequent runs can leave `auth_token` empty.
- Configure `mail.label` to the Gmail label you want to poll. Use `a.SetTestMode(true)` to use label `test`.

### Building

If you're using the `embedding` package with ONNX models, you'll need the tokenizers library:

```bash
./build.sh
```

This script automatically downloads and builds the required `libtokenizers.a` library if not present.

For standard builds without embeddings:

```bash
go mod tidy
go run .
```

If you maintain this repo/module directly and want an example `main.go` inside it, add it at the root of your app project rather than inside these packages.

### Production tips

- Prefer environment variables or a secret manager over committing keys to `config.yaml`
- Handle LLM API errors and timeouts robustly; consider retries and rate limits
- Validate Slack event types and signatures if you later move away from Socket Mode
- Persist `mail.maxid` (or store last processed message ID elsewhere) to avoid reprocessing

### Troubleshooting

- Slack client not connecting: verify App-Level Token, enable Socket Mode, and required scopes
- No events received: ensure Event Subscriptions include `app_mention` and message events; app is installed to the workspace and the channel
- LLM errors: verify API key and model name; watch for quota limits. Check that `gpt.provider` matches the key you supplied — a `claude-` model with an `sk-` key (or the reverse) will fail to authenticate
- Embedding errors on an Anthropic config: embeddings are OpenAI-only, see **Choosing an LLM provider**
- Gmail auth: ensure the token cache under `~/.credentials/` is created; re-run with `mail.auth_token` if needed

