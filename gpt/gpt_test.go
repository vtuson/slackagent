package gpt

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func newOpenAITestServer(t *testing.T, respond func(w http.ResponseWriter)) (*OpenAI, *map[string]interface{}, *http.Header) {
	t.Helper()

	var body map[string]interface{}
	var header http.Header

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		header = r.Header.Clone()
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("reading request body: %v", err)
		}
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("unmarshalling request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		respond(w)
	}))
	t.Cleanup(srv.Close)

	var o OpenAI
	o.SetApiKey("test-key")
	o.SetModel("gpt-3.5-turbo")
	o.SetURL(srv.URL)

	return &o, &body, &header
}

func openAIOK(text string) func(w http.ResponseWriter) {
	return func(w http.ResponseWriter) {
		w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"` + text + `"},"finish_reason":"stop"}]}`))
	}
}

func TestOpenAIGptQueryReturnsText(t *testing.T) {
	o, _, _ := newOpenAITestServer(t, openAIOK("a polite rewrite"))

	got, err := o.GptQuery("be polite", "you are wrong", "")
	if err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	if got != "a polite rewrite" {
		t.Errorf("got %q, want %q", got, "a polite rewrite")
	}
}

// OpenAI keeps the system prompt as a message role. This is the structural
// difference from Anthropic, so pin it.
func TestOpenAISystemPromptIsAMessage(t *testing.T) {
	o, body, _ := newOpenAITestServer(t, openAIOK("ok"))

	if _, err := o.GptQuery("be polite", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	if _, present := (*body)["system"]; present {
		t.Error("OpenAI requests must not carry a top-level system field")
	}

	messages := (*body)["messages"].([]interface{})
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages without context, got %d", len(messages))
	}
	first := messages[0].(map[string]interface{})
	if first["role"] != SYSTEMROLE || first["content"] != "be polite" {
		t.Errorf("first message = %#v, want a system message with the prompt", first)
	}
}

func TestOpenAIContextAddsThirdMessage(t *testing.T) {
	o, body, _ := newOpenAITestServer(t, openAIOK("ok"))

	if _, err := o.GptQuery("sys", "the message", "the context"); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	messages := (*body)["messages"].([]interface{})
	if len(messages) != 3 {
		t.Fatalf("expected 3 messages with context, got %d", len(messages))
	}
	third := messages[2].(map[string]interface{})
	if third["content"] != "the context" {
		t.Errorf("third message content = %v, want %q", third["content"], "the context")
	}
}

func TestOpenAISendsBearerAuth(t *testing.T) {
	o, _, header := newOpenAITestServer(t, openAIOK("ok"))

	if _, err := o.GptQuery("sys", "hello", ""); err != nil {
		t.Fatalf("GptQuery returned error: %v", err)
	}
	if got := header.Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("Authorization = %q, want %q", got, "Bearer test-key")
	}
}

func TestOpenAINoChoicesIsAnError(t *testing.T) {
	o, _, _ := newOpenAITestServer(t, func(w http.ResponseWriter) {
		w.Write([]byte(`{"choices":[]}`))
	})

	if _, err := o.GptQuery("sys", "hello", ""); err == nil {
		t.Fatal("expected an error when the response has no choices, got nil")
	}
}
