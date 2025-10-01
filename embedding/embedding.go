package embedding

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/vtuson/slackagent/gpt"
	ort "github.com/yalue/onnxruntime_go"
)

// EmbeddingProvider is an interface for getting embeddings from any provider (local or remote)
type EmbeddingProvider interface {
	GetEmbedding(text string) ([]float32, error)
	GetEmbeddingsBatch(texts []string) ([][]float32, error)
}

// LocalEmbeddingProvider implements EmbeddingProvider using a local ONNX model
type LocalEmbeddingProvider struct {
	pipeline *pipelines.FeatureExtractionPipeline
}

// OpenAIEmbeddingProvider implements EmbeddingProvider using OpenAI's API
type OpenAIEmbeddingProvider struct {
	*gpt.OpenAI
}

// GetEmbedding generates an embedding using the local ONNX model
func (lep *LocalEmbeddingProvider) GetEmbedding(text string) ([]float32, error) {
	if lep.pipeline == nil {
		return nil, fmt.Errorf("embedding pipeline not initialized")
	}

	batch := []string{text}
	result, err := lep.pipeline.RunPipeline(batch)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	if len(result.Embeddings) == 0 || len(result.Embeddings[0]) == 0 {
		return nil, fmt.Errorf("empty embedding result")
	}

	return result.Embeddings[0], nil
}

// GetEmbeddingsBatch generates embeddings for multiple texts using the local ONNX model
func (lep *LocalEmbeddingProvider) GetEmbeddingsBatch(texts []string) ([][]float32, error) {
	if lep.pipeline == nil {
		return nil, fmt.Errorf("embedding pipeline not initialized")
	}

	result, err := lep.pipeline.RunPipeline(texts)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embeddings: %w", err)
	}

	return result.Embeddings, nil
}

// EmbeddingDocument represents a chunk of text with its embedding
type EmbeddingDocument struct {
	ID        string    `json:"id"`
	Content   string    `json:"content"`
	URL       string    `json:"url"`
	Title     string    `json:"title"`
	Embedding []float32 `json:"embedding"`
	Depth     int       `json:"depth"`
}

// EmbeddingStore manages embeddings in memory and persists to file
type EmbeddingStore struct {
	Documents   []EmbeddingDocument `json:"documents"`
	LastUpdated string              `json:"last_updated"` // RFC3339 timestamp
	mu          sync.RWMutex
	filePath    string
	session     *hugot.Session
	pipeline    *pipelines.FeatureExtractionPipeline
	provider    EmbeddingProvider
}

// NewEmbeddingStore creates a new embedding store
func NewEmbeddingStore(filePath string, modelPath string, onnxLibPath string, provider EmbeddingProvider) (*EmbeddingStore, error) {
	// Create options with ONNX library path
	sessionOpts := []options.WithOption{}
	if onnxLibPath != "" {
		sessionOpts = append(sessionOpts, options.WithOnnxLibraryPath(onnxLibPath))
	}

	// Create a new hugot session
	session, err := hugot.NewORTSession(sessionOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create hugot session: %w", err)
	}
	log.Printf("Successfully created ORT session")

	// Create ONNX Runtime session options for model loading
	ortSessionOpts, err := ort.NewSessionOptions()
	if err != nil {
		session.Destroy()
		return nil, fmt.Errorf("failed to create ORT session options: %w", err)
	}
	defer ortSessionOpts.Destroy()

	// Create options for model loading with ORT backend
	opts := &options.Options{
		Backend:        "ORT",
		BackendOptions: ortSessionOpts,
		Destroy:        session.Destroy,
	}
	if onnxLibPath != "" {
		opts.ORTOptions = &options.OrtOptions{
			LibraryPath: &onnxLibPath,
		}
	}

	model, err := pipelineBackends.LoadModel(modelPath, "", opts)
	if err != nil {
		session.Destroy()
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	// Create the embedding pipeline configuration with normalization enabled
	config := pipelineBackends.PipelineConfig[*pipelines.FeatureExtractionPipeline]{
		ModelPath: modelPath,
		Name:      "embeddingsPipeline",
		Options: []pipelineBackends.PipelineOption[*pipelines.FeatureExtractionPipeline]{
			pipelines.WithNormalization(),
		},
	}

	// Initialize the pipeline
	pipeline, err := pipelines.NewFeatureExtractionPipeline(config, opts, model)
	if err != nil {
		session.Destroy()
		return nil, fmt.Errorf("failed to create embedding pipeline: %w", err)
	}

	localProvider := &LocalEmbeddingProvider{
		pipeline: pipeline,
	} // Create local embedding provider
	if provider == nil {
		provider = localProvider
	}

	return &EmbeddingStore{
		Documents: make([]EmbeddingDocument, 0),
		filePath:  filePath,
		session:   session,
		pipeline:  pipeline,
		provider:  provider,
	}, nil
}

// Destroy cleans up the embedding store runtime resources (hugot session/pipeline)
// Note: This does NOT delete the persisted embeddings file - only frees memory
func (es *EmbeddingStore) Destroy() {
	if es.session != nil {
		es.session.Destroy()
	}
}

// Load loads embeddings from file
func (es *EmbeddingStore) Load() error {
	es.mu.Lock()
	defer es.mu.Unlock()

	data, err := os.ReadFile(es.filePath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Printf("Embeddings file does not exist, starting fresh")
			return nil
		}
		return fmt.Errorf("failed to read embeddings file: %w", err)
	}

	if err := json.Unmarshal(data, es); err != nil {
		return fmt.Errorf("failed to unmarshal embeddings: %w", err)
	}

	log.Printf("Loaded %d embeddings from %s", len(es.Documents), es.filePath)
	return nil
}

// Save persists embeddings to file
func (es *EmbeddingStore) Save() error {
	es.mu.Lock()
	defer es.mu.Unlock()

	// Update timestamp
	es.LastUpdated = time.Now().Format(time.RFC3339)

	data, err := json.MarshalIndent(es, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal embeddings: %w", err)
	}

	if err := os.WriteFile(es.filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write embeddings file: %w", err)
	}

	log.Printf("Saved %d embeddings to %s", len(es.Documents), es.filePath)
	return nil
}

// AddDocument adds a document with its embedding
func (es *EmbeddingStore) AddDocument(doc EmbeddingDocument) {
	es.mu.Lock()
	defer es.mu.Unlock()
	es.Documents = append(es.Documents, doc)
}

// Clear removes all documents
func (es *EmbeddingStore) Clear() {
	es.mu.Lock()
	defer es.mu.Unlock()
	es.Documents = make([]EmbeddingDocument, 0)
	es.LastUpdated = ""
}

// IsStale checks if embeddings are older than the specified number of days
func (es *EmbeddingStore) IsStale(days int) bool {
	es.mu.RLock()
	defer es.mu.RUnlock()

	if es.LastUpdated == "" {
		return true
	}

	lastUpdate, err := time.Parse(time.RFC3339, es.LastUpdated)
	if err != nil {
		log.Printf("Failed to parse last_updated timestamp: %v", err)
		return true
	}

	age := time.Since(lastUpdate)
	staleThreshold := time.Duration(days) * 24 * time.Hour

	return age > staleThreshold
}

// normalizeEmbedding performs L2 normalization on an embedding vector
func NormalizeEmbedding(embedding []float32) []float32 {
	var sum float32
	for _, v := range embedding {
		sum += v * v
	}
	norm := float32(math.Sqrt(float64(sum)))

	if norm == 0 {
		return embedding
	}

	normalized := make([]float32, len(embedding))
	for i, v := range embedding {
		normalized[i] = v / norm
	}
	return normalized
}

// SetProvider sets a custom embedding provider (e.g., GPT-based)
func (es *EmbeddingStore) SetProvider(provider EmbeddingProvider) {
	es.mu.Lock()
	defer es.mu.Unlock()
	es.provider = provider
}

// GetEmbedding generates an embedding for the given text using the configured provider
func (es *EmbeddingStore) GetEmbedding(ctx context.Context, text string) ([]float32, error) {
	if es.provider == nil {
		return nil, fmt.Errorf("embedding provider not initialized")
	}
	return es.provider.GetEmbedding(text)
}

// GetEmbeddingsBatch generates embeddings for multiple texts at once using the configured provider
func (es *EmbeddingStore) GetEmbeddingsBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if es.provider == nil {
		return nil, fmt.Errorf("embedding provider not initialized")
	}
	return es.provider.GetEmbeddingsBatch(texts)
}

// CosineSimilarity calculates the cosine similarity between two vectors
// Note: Assumes vectors are already normalized (unit length)
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0.0
	}

	// For normalized vectors, cosine similarity = dot product
	var dotProduct float32
	for i := range a {
		dotProduct += a[i] * b[i]
	}

	return dotProduct
}

// SearchResult represents a search result with similarity score
type SearchResult struct {
	Document   EmbeddingDocument
	Similarity float32
}

// Search finds the most similar documents to the query
func (es *EmbeddingStore) Search(ctx context.Context, query string, topK int, threshold float32) ([]SearchResult, error) {
	es.mu.RLock()
	defer es.mu.RUnlock()

	if len(es.Documents) == 0 {
		return nil, nil
	}

	// Get embedding for query
	queryEmbedding, err := es.GetEmbedding(ctx, query)
	if err != nil {
		return nil, err
	}
	log.Printf("Query embedding dimension: %d", len(queryEmbedding))
	log.Printf("Document embedding dimension: %d", len(es.Documents[0].Embedding))

	// Check if query embedding is normalized
	var queryNorm float32
	for _, v := range queryEmbedding {
		queryNorm += v * v
	}
	log.Printf("Query embedding norm: %.6f (should be 1.0)", math.Sqrt(float64(queryNorm)))

	// Log first few values of query embedding for debugging
	if len(queryEmbedding) > 5 {
		log.Printf("Query embedding sample: [%.4f, %.4f, %.4f, %.4f, %.4f]",
			queryEmbedding[0], queryEmbedding[1], queryEmbedding[2], queryEmbedding[3], queryEmbedding[4])
	}
	if len(es.Documents[0].Embedding) > 5 {
		log.Printf("Doc embedding sample: [%.4f, %.4f, %.4f, %.4f, %.4f]",
			es.Documents[0].Embedding[0], es.Documents[0].Embedding[1], es.Documents[0].Embedding[2],
			es.Documents[0].Embedding[3], es.Documents[0].Embedding[4])
	}

	// Calculate similarities
	results := make([]SearchResult, 0)
	allScores := make([]SearchResult, 0) // Keep all scores for debugging

	for _, doc := range es.Documents {
		similarity := CosineSimilarity(queryEmbedding, doc.Embedding)

		searchResult := SearchResult{
			Document:   doc,
			Similarity: similarity,
		}
		allScores = append(allScores, searchResult)

		if similarity >= threshold {
			results = append(results, searchResult)
		}
	}

	// Sort all scores for debugging (descending)
	for i := 0; i < len(allScores); i++ {
		for j := i + 1; j < len(allScores); j++ {
			if allScores[j].Similarity > allScores[i].Similarity {
				allScores[i], allScores[j] = allScores[j], allScores[i]
			}
		}
	}

	// Log top 5 similarity scores for debugging
	log.Printf("Query: %s", query)
	log.Printf("Top similarity scores:")
	for i := 0; i < len(allScores) && i < 5; i++ {
		contentPreview := allScores[i].Document.Content
		if len(contentPreview) > 100 {
			contentPreview = contentPreview[:100] + "..."
		}
		log.Printf("  [%d] Score: %.4f - %s", i+1, allScores[i].Similarity, contentPreview)
	}

	// Sort results by similarity (descending)
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Similarity > results[i].Similarity {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Return top K results
	if topK > 0 && len(results) > topK {
		results = results[:topK]
	}

	return results, nil
}

// ChunkText splits text into smaller chunks for embedding
// This helps with token limits and more precise retrieval
func ChunkText(text string, chunkSize int, overlap int) []string {
	// Clean and normalize text
	text = strings.TrimSpace(text)
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")

	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}

	chunks := make([]string, 0)
	for i := 0; i < len(words); i += chunkSize - overlap {
		end := i + chunkSize
		if end > len(words) {
			end = len(words)
		}
		chunk := strings.Join(words[i:end], " ")
		chunks = append(chunks, chunk)
		if end >= len(words) {
			break
		}
	}

	return chunks
}

// ExtractLinks extracts page URLs from content based on the base URL pattern
// baseURL should be the domain pattern to match (e.g., "https://www.notion.so/")
func ExtractLinks(content string, baseURL string) []string {
	// Escape special regex characters in baseURL and create pattern
	escapedBase := regexp.QuoteMeta(baseURL)
	pattern := escapedBase + `[a-zA-Z0-9\-]+`

	re := regexp.MustCompile(pattern)
	matches := re.FindAllString(content, -1)

	// Deduplicate
	seen := make(map[string]bool)
	links := make([]string, 0)
	for _, match := range matches {
		if !seen[match] {
			seen[match] = true
			links = append(links, match)
		}
	}

	return links
}
