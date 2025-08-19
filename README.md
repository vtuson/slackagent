### GPT Slack Agent scaffolding

An opinionated Go scaffolding to build Slack agents powered by LLMs. It wires together:

- **Slack Socket Mode** event ingestion and posting utilities
- **OpenAI Chat Completions** helper
- Optional **Gmail** polling utilities for email-driven workflows

You bring a small `main.go` that plugs in your business logic (Slack and/or Mail processors). This package provides the plumbing so you can focus on your agent behavior.

### Features

- **Socket Mode Slack client** with helpers to post to channels/threads, fetch thread replies, post/remove reactions and basic text formatting
- **Event filter** that forwards `app_mention` and plain `message` events for your processing
- **OpenAI client** wrapper with a simple `GptQuery` API and sensible defaults
- **Gmail** utilities for polling labeled messages and parsing bodies (plain and HTML)
- **MCP client** with support for Streamable, SSE, and STDIO transports for Model Context Protocol integration
- **Notion MCP** integration with specialized client for Notion's MCP implementation
- **YAML config** loader, including pass-through `agent_config` for your custom settings

### Repository layout

- `agent/` — Core agent wiring: config loader, Slack client initialization, email loop, LLM factory, MCP integration
  - `mcp.go` — Model Context Protocol client implementation with multiple transport options
  - `notionmcp.go` — Specialized Notion MCP client implementation
  - `headers.go` — HTTP header utilities for MCP clients
- `slack/` — Slack client and helpers (`PostInChannel`, `PostInThread`, `GetThreadMessages`, `StripAtMention`, `AddText`)
- `gpt/` — Minimal OpenAI Chat Completions helper
- `mail/` — Gmail connection and parsing utils
- `config.yaml` — Example configuration

### Requirements

- Go 1.20+
- A Slack app with Bot Token and App-Level Token (Socket Mode)
- OpenAI API key
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
  key: "sk-..."           # OpenAI API Key
  model: "gpt-3.5-turbo"  # Model name

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

### Programmatic APIs

- `agent.Agent`
  - `LoadConfig(path string)` — read YAML
  - `InitializeSlackClient()` — start Socket Mode client
  - `GetSlackClient() *slack.Client` — access Slack helpers
  - `HasEmail() bool` / `ProcessEmails()` — start Gmail polling loop
  - `NewLLM() *gpt.OpenAI` — build LLM client with configured model/key
  - `GetCustomConfig(out interface{}) error` — unmarshal `agent_config` into your struct

- `slack.Client`
  - `PostInChannel(channel, message)` and `PostInThread(channel, message, threadTS)`
  - `GetThreadMessages(channel, threadTS)`
  - `AddText(text, to, bold)` and `StripAtMention(text)`

- `gpt.OpenAI`
  - `SetApiKey`, `SetModel`
  - `GptQuery(systemPrompt, message, context)` returns the assistant reply string

- `mail` utilities
  - `Connect(secretPath, code)`; `GetEmails(...)`; `Email.Text()`; HTML/text decode helpers

- `agent.MCPClient`
  - `ConnectMCP(ctx, opts)` — Connect to an MCP server with various transport options
  - `ListTools(ctx)` — List all available tools from the MCP server
  - `CallTool(ctx, name, arguments)` — Call a specific tool with arguments
  - `ExtractTextResponses(result)` — Helper to extract text responses from tool results

- `agent.NotionMCP`
  - `Streamable()` — Connect to Notion MCP using Streamable transport
  - `STDIO()` — Connect to Notion MCP using STDIO transport
  - `STDIOStreamable()` — Connect to Notion MCP using STDIO with Streamable transport

### Running locally

```bash
go mod tidy
go run .
```

If you maintain this repo/module directly and want an example `main.go` inside it, add it at the root of your app project rather than inside these packages.

### Production tips

- Prefer environment variables or a secret manager over committing keys to `config.yaml`
- Handle OpenAI/API errors and timeouts robustly; consider retries and rate limits
- Validate Slack event types and signatures if you later move away from Socket Mode
- Persist `mail.maxid` (or store last processed message ID elsewhere) to avoid reprocessing

### Troubleshooting

- Slack client not connecting: verify App-Level Token, enable Socket Mode, and required scopes
- No events received: ensure Event Subscriptions include `app_mention` and message events; app is installed to the workspace and the channel
- OpenAI errors: verify API key and model name; watch for quota limits
- Gmail auth: ensure the token cache under `~/.credentials/` is created; re-run with `mail.auth_token` if needed

