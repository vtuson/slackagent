package agent

import (
	"net/http"
)

type headerTransport struct {
	base    http.RoundTripper
	headers map[string]string
}

func (h *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Clone the request to avoid modifying the original
	reqClone := req.Clone(req.Context())

	// Add all configured headers
	for key, value := range h.headers {
		reqClone.Header.Set(key, value)
	}

	// Use the base transport to make the actual request
	return h.base.RoundTrip(reqClone)
}

// Helper function to create the transport
func newHeaderTransport(base http.RoundTripper, headers map[string]string) *headerTransport {
	if base == nil {
		base = http.DefaultTransport
	}
	return &headerTransport{
		base:    base,
		headers: headers,
	}
}
