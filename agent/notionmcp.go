package agent

import (
	"context"
	"net/http"
)

const (
	NotionMCPURL = "https://mcp.notion.com/mcp"
)

type NotionMCP struct {
	NotionKey             string
	ImplementationName    string
	ImplementationVersion string
	URL                   string
}

// not clear this is working correctly... still getting 401 errors
func (n *NotionMCP) Streamable() (*MCPClient, error) {
	if n.URL == "" {
		n.URL = NotionMCPURL
	}

	// Create client with custom headers
	client := &http.Client{
		Transport: &headerTransport{
			base: http.DefaultTransport,
			headers: map[string]string{
				"Authorization":  "Bearer " + n.NotionKey,
				"Content-Type":   "application/json",
				"Notion-Version": "2022-06-28",
			},
		},
	}

	mcpClient, err := ConnectMCP(context.Background(), MCPOptions{
		ImplementationName:    n.ImplementationName,
		ImplementationVersion: n.ImplementationVersion,
		Method:                MethodStreamable,
		URL:                   n.URL,
		Args:                  []string{},
		HTTPClient:            client,
	})

	return mcpClient, err
}

func (n *NotionMCP) STDIO() (*MCPClient, error) {
	if n.URL == "" {
		n.URL = NotionMCPURL
	}
	// this method requires manual validation of access to the Notion API
	mcpClient, err := ConnectMCP(context.Background(), MCPOptions{
		ImplementationName:    n.ImplementationName,
		ImplementationVersion: n.ImplementationVersion,
		Method:                MethodSTDIO,
		Command:               "npx",
		Args:                  []string{"-y", "mcp-remote", n.URL},
		HTTPClient:            nil,
	})

	return mcpClient, err
}

// requires npx in the system to work correctly
func (n *NotionMCP) STDIOStreamable() (*MCPClient, error) {

	mcpClient, err := ConnectMCP(context.Background(), MCPOptions{
		ImplementationName:    n.ImplementationName,
		ImplementationVersion: n.ImplementationVersion,
		Method:                MethodSTDIO,
		Command:               "npx",
		Args:                  []string{"-y", "notionhq/notion-mcp-server", "--transport", "http", "--auth-token", n.NotionKey},
		HTTPClient:            nil,
	})

	return mcpClient, err
}
