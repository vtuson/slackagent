package agent

import (
	"testing"

	"github.com/vtuson/slackagent/gpt"
)

// newTestAgent builds an Agent with just the GPT config populated, bypassing
// LoadConfig (which exits the process when Slack tokens are absent).
func newTestAgent(key, model, provider string) *Agent {
	cfg := &Config{}
	cfg.GPT = &struct {
		Key      string `yaml:"key"`
		Model    string `yaml:"model"`
		Provider string `yaml:"provider,omitempty"`
	}{Key: key, Model: model, Provider: provider}

	return &Agent{Config: cfg}
}

func TestNewLLMExplicitProvider(t *testing.T) {
	a := newTestAgent("k", "claude-opus-5", gpt.PROVIDERANTHROPIC)
	if _, ok := a.NewLLM().(*gpt.Claude); !ok {
		t.Errorf("expected *gpt.Claude for an anthropic provider, got %T", a.NewLLM())
	}

	a = newTestAgent("k", "gpt-3.5-turbo", gpt.PROVIDEROPENAI)
	if _, ok := a.NewLLM().(*gpt.OpenAI); !ok {
		t.Errorf("expected *gpt.OpenAI for an openai provider, got %T", a.NewLLM())
	}
}

// Configs written before the provider field existed must keep working.
func TestNewLLMBackwardsCompatible(t *testing.T) {
	a := newTestAgent("k", "gpt-3.5-turbo", "")
	if _, ok := a.NewLLM().(*gpt.OpenAI); !ok {
		t.Errorf("a config with no provider and a gpt model must stay on OpenAI, got %T", a.NewLLM())
	}
}

// A claude model with no provider set should be inferred.
func TestNewLLMInfersAnthropic(t *testing.T) {
	a := newTestAgent("k", "claude-haiku-4-5", "")
	if _, ok := a.NewLLM().(*gpt.Claude); !ok {
		t.Errorf("a claude model should infer the anthropic provider, got %T", a.NewLLM())
	}
}

// An empty model must default per provider, not to the OpenAI default.
func TestNewLLMDefaultsModelPerProvider(t *testing.T) {
	a := newTestAgent("k", "", gpt.PROVIDERANTHROPIC)
	a.NewLLM()
	if a.Config.GPT.Model != gpt.MODELCLAUDE {
		t.Errorf("model = %q, want %q", a.Config.GPT.Model, gpt.MODELCLAUDE)
	}

	a = newTestAgent("k", "", gpt.PROVIDEROPENAI)
	a.NewLLM()
	if a.Config.GPT.Model != gpt.GetDefaultModel() {
		t.Errorf("model = %q, want %q", a.Config.GPT.Model, gpt.GetDefaultModel())
	}
}

func TestNewEmbedderRejectsAnthropic(t *testing.T) {
	a := newTestAgent("k", "claude-opus-5", gpt.PROVIDERANTHROPIC)
	if _, err := a.NewEmbedder(); err == nil {
		t.Fatal("expected an error: anthropic has no embeddings endpoint")
	}

	a = newTestAgent("k", "gpt-3.5-turbo", gpt.PROVIDEROPENAI)
	if _, err := a.NewEmbedder(); err != nil {
		t.Fatalf("openai embedder should work: %v", err)
	}
}
