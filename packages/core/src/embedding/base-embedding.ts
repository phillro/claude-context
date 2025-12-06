// Interface definitions
export interface EmbeddingVector {
    vector: number[];
    dimension: number;
}

/**
 * Abstract base class for embedding implementations
 */
export abstract class Embedding {
    protected abstract maxTokens: number;

    /**
     * Preprocess text to ensure it's valid for embedding
     * @param text Input text
     * @returns Processed text
     */
    protected preprocessText(text: string): string {
        // Replace empty string with single space
        if (text === '') {
            return ' ';
        }

        // Simple character-based truncation (approximation)
        // For code, tokens are shorter than English prose (~3 chars/token vs ~4 chars/token)
        // Use 3 chars/token to be safe, with additional 10% safety margin for BPE edge cases
        const maxChars = Math.floor(this.maxTokens * 3 * 0.9);
        if (text.length > maxChars) {
            console.warn(`[Embedding] ⚠️ Truncating text from ${text.length} to ${maxChars} chars (maxTokens=${this.maxTokens})`);
            return text.substring(0, maxChars);
        }

        return text;
    }

    /**
     * Detect embedding dimension 
     * @param testText Test text for dimension detection
     * @returns Embedding dimension
     */
    abstract detectDimension(testText?: string): Promise<number>;

    /**
     * Preprocess array of texts
     * @param texts Array of input texts
     * @returns Array of processed texts
     */
    protected preprocessTexts(texts: string[]): string[] {
        return texts.map(text => this.preprocessText(text));
    }

    // Abstract methods that must be implemented by subclasses
    /**
     * Generate text embedding vector
     * @param text Text content
     * @returns Embedding vector
     */
    abstract embed(text: string): Promise<EmbeddingVector>;

    /**
     * Generate text embedding vectors in batch
     * @param texts Text array
     * @returns Embedding vector array
     */
    abstract embedBatch(texts: string[]): Promise<EmbeddingVector[]>;

    /**
     * Get embedding vector dimension
     * @returns Vector dimension
     */
    abstract getDimension(): number;

    /**
     * Get service provider name
     * @returns Provider name
     */
    abstract getProvider(): string;
} 