package agent

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// MCPConnectionMethod identifies how to connect to an MCP server.
type MCPConnectionMethod string

const (
	// MethodRemote uses a local stdio proxy (npx mcp-remote) to connect to a remote MCP URL.
	MethodRemote MCPConnectionMethod = "remote"
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
	Command string
	Args    []string
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
		opts.ImplementationName = "slackagent"
	}
	if opts.ImplementationVersion == "" {
		opts.ImplementationVersion = "v0.0.1"
	}

	client := mcp.NewClient(&mcp.Implementation{
		Name:    opts.ImplementationName,
		Version: opts.ImplementationVersion,
	}, nil)

	var transport mcp.Transport

	switch opts.Method {
	case MethodRemote:
		if opts.URL == "" {
			return nil, errors.New("remote method selected but URL is empty")
		}
		// Use Notion’s recommended approach: a local stdio proxy via npx mcp-remote
		// Ref: https://developers.notion.com/docs/get-started-with-mcp#connect-through-your-ai-tool
		cmd := exec.Command("npx", append([]string{"-y", "mcp-remote", opts.URL}, opts.Args...)...)
		transport = mcp.NewCommandTransport(cmd)

	case MethodSTDIO:
		if opts.Command == "" {
			return nil, errors.New("stdio method selected but Command is empty")
		}
		cmd := exec.Command(opts.Command, opts.Args...)
		transport = mcp.NewCommandTransport(cmd)

	default:
		return nil, fmt.Errorf("unsupported MCP connection method: %q", string(opts.Method))
	}

	session, err := client.Connect(ctx, transport)
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
