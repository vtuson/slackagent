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
- **Gmail** utilities for polling labeled messages and parsing bodies (plain and HTML)
- **MCP client** with support for Streamable, SSE, and STDIO transports for Model Context Protocol integration
- **Notion MCP** integration with specialized client for Notion's MCP implementation
- **Embedding utilities** with support for local ONNX models and OpenAI embeddings for RAG systems
- **YAML config** loader, including pass-through `agent_config` for your custom settings

### Repository layout

- `agent/` — Core agent wiring: config loader, Slack client initialization, email loop, LLM factory, MCP integration
  - `mcp.go` — Model Context Protocol client implementation with multiple transport options
  - `notionmcp.go` — Specialized Notion MCP client implementation
  - `headers.go` — HTTP header utilities for MCP clients
- `slack/` — Slack client and helpers (`PostInChannel`, `PostInThread`, `GetThreadMessages`, `StripAtMention`, `AddText`)
- `gpt/` — LLM helpers behind a common `LLM` interface (`GptQuery`, `Query`)
  - `llm.go` — The `LLM` interface, `Response` type, stop reasons, and the provider factory
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

> **Note:** `Query` is single-turn — it does not carry message history. The loop above feeds the text so far back through the `context` argument, which is an approximation of a real continuation rather than resuming an assistant turn. It is enough to detect and react to truncation; if you need true multi-turn continuation, raise an issue and the interface can grow a history-carrying call.

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

