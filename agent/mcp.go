package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/vtuson/slackagent/gpt"
)

// MCPConnectionMethod identifies how to connect to an MCP server.
type MCPConnectionMethod string

const (
	// MethodStreamable connects to a remote MCP server using the Streamable HTTP transport.
	MethodStreamable MCPConnectionMethod = "streamable"
	// MethodSSE connects to a remote MCP server using SSE transport.
	MethodSSE MCPConnectionMethod = "sse"
	// MethodSTDIO launches a local MCP server process and connects over stdio.
	MethodSTDIO MCPConnectionMethod = "stdio"
)

// MCPOptions configures how to connect to an MCP server.
type MCPOptions struct {
	// Implementation metadata for the MCP client.
	ImplementationName    string
	ImplementationVersion string

	// Method selects the connection approach (remote via npx mcp-remote or stdio).
	Method MCPConnectionMethod

	// Remote URL (used when MethodRemote). Example: https://mcp.notion.com/mcp
	URL string

	// Local command and args (used when MethodSTDIO). Example: command="myserver", args=["--flag"]
	Command    string
	Args       []string
	Env        []string
	HTTPClient *http.Client
}

// MCPClient wraps an MCP client session and provides convenience helpers.
type MCPClient struct {
	client  *mcp.Client
	session *mcp.ClientSession
	mu      sync.RWMutex
}

// ConnectMCP connects to an MCP server using the provided options and returns a ready session.
func ConnectMCP(ctx context.Context, opts MCPOptions) (*MCPClient, error) {
	if opts.ImplementationName == "" {
		return nil, errors.New("implementation name is required")
	}
	if opts.ImplementationVersion == "" {
		opts.ImplementationVersion = "v0.0.1"
		log.Println("implementation version is empty, using default")
	}

	client := mcp.NewClient(&mcp.Implementation{
		Name:    opts.ImplementationName,
		Version: opts.ImplementationVersion,
	}, nil)

	var transport mcp.Transport

	switch opts.Method {
	case MethodStreamable:
		if opts.URL == "" {
			return nil, errors.New("streamable method selected but URL is empty")
		}
		log.Printf("connecting to MCP server with URL: %s", opts.URL)
		transport = &mcp.StreamableClientTransport{
			Endpoint:   opts.URL,
			HTTPClient: opts.HTTPClient,
		}
	case MethodSSE:
		if opts.URL == "" {
			return nil, errors.New("sse method selected but URL is empty")
		}
		transport = &mcp.SSEClientTransport{
			Endpoint:   opts.URL,
			HTTPClient: opts.HTTPClient,
		}
	case MethodSTDIO:
		if opts.Command == "" {
			return nil, errors.New("stdio method selected but Command is empty")
		}
		log.Printf("connecting to MCP server with command: %s and args: %v", opts.Command, opts.Args)
		cmd := exec.Command(opts.Command, opts.Args...)
		if opts.Env != nil {
			for _, env := range opts.Env {
				cmd.Env = append(os.Environ(), env)
			}
		}
		transport = &mcp.CommandTransport{Command: cmd}

	default:
		return nil, fmt.Errorf("unsupported MCP connection method: %q", string(opts.Method))
	}

	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return nil, fmt.Errorf("connect MCP: %w", err)
	}

	return &MCPClient{client: client, session: session}, nil
}

// Close closes the underlying MCP session and releases resources.
func (c *MCPClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.session == nil {
		return nil
	}
	err := c.session.Close()
	c.session = nil
	c.client = nil
	return err
}

// ListTools retrieves all tools exposed by the connected MCP server.
// It transparently handles pagination if the server returns a cursor.
func (c *MCPClient) ListTools(ctx context.Context) ([]*mcp.Tool, error) {
	c.mu.RLock()
	session := c.session
	c.mu.RUnlock()
	if session == nil {
		return nil, errors.New("mcp session is not connected")
	}

	var all []*mcp.Tool
	var cursor string

	for {
		// The SDK exposes a typed ListTools API; loop until no next cursor.
		res, err := session.ListTools(ctx, &mcp.ListToolsParams{Cursor: cursor})
		if err != nil {
			return nil, fmt.Errorf("list tools: %w", err)
		}
		all = append(all, res.Tools...)
		if res.NextCursor == "" {
			break
		}
		cursor = res.NextCursor
	}
	return all, nil
}

// CallTool invokes a tool by name with the provided arguments.
// Arguments should match the tool’s JSON Schema.
func (c *MCPClient) CallTool(ctx context.Context, name string, arguments map[string]any) (*mcp.CallToolResult, error) {
	c.mu.RLock()
	session := c.session
	c.mu.RUnlock()
	if session == nil {
		return nil, errors.New("mcp session is not connected")
	}

	params := &mcp.CallToolParams{
		Name:      name,
		Arguments: arguments,
	}
	res, err := session.CallTool(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("call tool %q: %w", name, err)
	}
	return res, nil
}

// ExtractTextResponses is a helper to convert a CallToolResult’s content to plain text strings
// when the tool returns textual responses.
func ExtractTextResponses(res *mcp.CallToolResult) []string {
	if res == nil || len(res.Content) == 0 {
		return nil
	}
	texts := make([]string, 0, len(res.Content))
	for _, c := range res.Content {
		if tc, ok := c.(*mcp.TextContent); ok {
			texts = append(texts, tc.Text)
		}
	}
	return texts
}

// Tools lists the server's tools in the provider-neutral form the gpt package
// takes, ready to hand to LLM.NewChat. The JSON Schema each server publishes
// is passed through untouched, so the model sees the same argument contract
// the server will validate against.
func (c *MCPClient) Tools(ctx context.Context) ([]gpt.Tool, error) {
	tools, err := c.ListTools(ctx)
	if err != nil {
		return nil, err
	}

	out := make([]gpt.Tool, 0, len(tools))
	for _, t := range tools {
		schema, err := schemaToMap(t.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("tool %q has an unreadable input schema: %w", t.Name, err)
		}
		out = append(out, gpt.Tool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: schema,
		})
	}
	return out, nil
}

// schemaToMap normalises a published schema into a map. The MCP client types
// it as any, and what arrives depends on the transport, so it goes through
// JSON rather than a type assertion.
func schemaToMap(schema any) (map[string]any, error) {
	if schema == nil {
		return nil, nil
	}
	if m, ok := schema.(map[string]any); ok {
		return m, nil
	}

	raw, err := json.Marshal(schema)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	return m, nil
}

// ToolRunner returns a runner that dispatches the model's tool calls to this
// server, for use with gpt.RunToolLoop.
func (c *MCPClient) ToolRunner(ctx context.Context) gpt.ToolFunc {
	return func(call gpt.ToolCall) gpt.ToolResult {
		return c.RunTool(ctx, call)
	}
}

// RunTool runs one tool call and packages the outcome for the model.
//
// Failures come back as a result marked IsError rather than as a Go error: the
// model can read "no such page" and try something else, which it cannot do if
// the conversation stops. Only a broken session is worth aborting for, and
// that shows up on the next turn anyway.
func (c *MCPClient) RunTool(ctx context.Context, call gpt.ToolCall) gpt.ToolResult {
	args, err := call.Arguments()
	if err != nil {
		return gpt.ToolResult{ID: call.ID, Content: err.Error(), IsError: true}
	}

	res, err := c.CallTool(ctx, call.Name, args)
	if err != nil {
		return gpt.ToolResult{ID: call.ID, Content: err.Error(), IsError: true}
	}

	content := strings.Join(ExtractTextResponses(res), "\n")
	if content == "" && res.StructuredContent != nil {
		// A tool that only returns structured output has no text
		// blocks to join, so hand the model the JSON.
		if raw, err := json.Marshal(res.StructuredContent); err == nil {
			content = string(raw)
		}
	}
	if content == "" {
		// An empty result reads as a broken tool to the model, so say
		// plainly that it did run.
		content = "the tool returned no output"
	}

	return gpt.ToolResult{ID: call.ID, Content: content, IsError: res.IsError}
}
