import OpenAI from 'openai';
import { Embedding, EmbeddingVector } from './base-embedding';

export interface OpenAIEmbeddingConfig {
    model: string;
    apiKey: string;
    baseURL?: string; // OpenAI supports custom baseURL
    maxTokens?: number; // Override max tokens for custom models (default varies by model)
}

export class OpenAIEmbedding extends Embedding {
    private client: OpenAI;
    private config: OpenAIEmbeddingConfig;
    private dimension: number = 1536; // Default dimension for text-embedding-3-small
    protected maxTokens: number = 8192; // Maximum tokens for OpenAI embedding models

    constructor(config: OpenAIEmbeddingConfig) {
        super();
        this.config = config;
        this.client = new OpenAI({
            apiKey: config.apiKey,
            baseURL: config.baseURL,
        });

        // Set maxTokens from config or known models
        if (config.maxTokens) {
            this.maxTokens = config.maxTokens;
        } else {
            const knownModels = OpenAIEmbedding.getSupportedModels();
            if (knownModels[config.model]) {
                this.maxTokens = knownModels[config.model].maxTokens;
                this.dimension = knownModels[config.model].dimension;
            }
        }
    }

    async detectDimension(testText: string = "test"): Promise<number> {
        const model = this.config.model || 'text-embedding-3-small';
        const knownModels = OpenAIEmbedding.getSupportedModels();

        // Use known dimension for standard models
        if (knownModels[model]) {
            return knownModels[model].dimension;
        }

        // For custom models, make API call to detect dimension
        try {
            const processedText = this.preprocessText(testText);
            const response = await this.client.embeddings.create({
                model: model,
                input: processedText,
                encoding_format: 'float',
            });
            return response.data[0].embedding.length;
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';

            // Re-throw authentication errors
            if (errorMessage.includes('API key') || errorMessage.includes('unauthorized') || errorMessage.includes('authentication')) {
                throw new Error(`Failed to detect dimension for model ${model}: ${errorMessage}`);
            }

            // For other errors, throw exception instead of using fallback
            throw new Error(`Failed to detect dimension for model ${model}: ${errorMessage}`);
        }
    }

    async embed(text: string): Promise<EmbeddingVector> {
        const processedText = this.preprocessText(text);
        const model = this.config.model || 'text-embedding-3-small';

        const knownModels = OpenAIEmbedding.getSupportedModels();
        if (knownModels[model] && this.dimension !== knownModels[model].dimension) {
            this.dimension = knownModels[model].dimension;
        } else if (!knownModels[model]) {
            this.dimension = await this.detectDimension();
        }

        try {
            const response = await this.client.embeddings.create({
                model: model,
                input: processedText,
                encoding_format: 'float',
            });

            // Update dimension from actual response
            this.dimension = response.data[0].embedding.length;

            return {
                vector: response.data[0].embedding,
                dimension: this.dimension
            };
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            throw new Error(`Failed to generate OpenAI embedding: ${errorMessage}`);
        }
    }

    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> {
        const processedTexts = this.preprocessTexts(texts);
        const model = this.config.model || 'text-embedding-3-small';

        const knownModels = OpenAIEmbedding.getSupportedModels();
        if (knownModels[model] && this.dimension !== knownModels[model].dimension) {
            this.dimension = knownModels[model].dimension;
        } else if (!knownModels[model]) {
            this.dimension = await this.detectDimension();
        }

        try {
            const response = await this.client.embeddings.create({
                model: model,
                input: processedTexts,
                encoding_format: 'float',
            });

            this.dimension = response.data[0].embedding.length;

            return response.data.map((item) => ({
                vector: item.embedding,
                dimension: this.dimension
            }));
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
            throw new Error(`Failed to generate OpenAI batch embeddings: ${errorMessage}`);
        }
    }

    getDimension(): number {
        // For custom models, we need to detect the dimension first
        const model = this.config.model || 'text-embedding-3-small';
        const knownModels = OpenAIEmbedding.getSupportedModels();

        // If it's a known model, return its known dimension
        if (knownModels[model]) {
            return knownModels[model].dimension;
        }

        // For custom models, return the current dimension
        // Note: This may be incorrect until detectDimension() is called
        console.warn(`[OpenAIEmbedding] ⚠️ getDimension() called for custom model '${model}' - returning ${this.dimension}. Call detectDimension() first for accurate dimension.`);
        return this.dimension;
    }

    getProvider(): string {
        return 'OpenAI';
    }

    /**
     * Set model type
     * @param model Model name
     */
    async setModel(model: string): Promise<void> {
        this.config.model = model;
        const knownModels = OpenAIEmbedding.getSupportedModels();
        if (knownModels[model]) {
            this.dimension = knownModels[model].dimension;
        } else {
            this.dimension = await this.detectDimension();
        }
    }

    /**
     * Get client instance (for advanced usage)
     */
    getClient(): OpenAI {
        return this.client;
    }

    /**
     * Get list of supported models with their dimensions and max token limits
     */
    static getSupportedModels(): Record<string, { dimension: number; maxTokens: number; description: string }> {
        return {
            // OpenAI models
            'text-embedding-3-small': {
                dimension: 1536,
                maxTokens: 8192,
                description: 'High performance and cost-effective embedding model (recommended)'
            },
            'text-embedding-3-large': {
                dimension: 3072,
                maxTokens: 8192,
                description: 'Highest performance embedding model with larger dimensions'
            },
            'text-embedding-ada-002': {
                dimension: 1536,
                maxTokens: 8192,
                description: 'Legacy model (use text-embedding-3-small instead)'
            },
            // BGE models (commonly used with vLLM)
            'BAAI/bge-large-en-v1.5': {
                dimension: 1024,
                maxTokens: 512,
                description: 'BGE large English embedding model (512 token limit)'
            },
            'BAAI/bge-base-en-v1.5': {
                dimension: 768,
                maxTokens: 512,
                description: 'BGE base English embedding model (512 token limit)'
            },
            'BAAI/bge-small-en-v1.5': {
                dimension: 384,
                maxTokens: 512,
                description: 'BGE small English embedding model (512 token limit)'
            },
            'BAAI/bge-m3': {
                dimension: 1024,
                maxTokens: 8192,
                description: 'BGE M3 multilingual embedding model (8192 token limit)'
            },
            // Nomic models
            'nomic-ai/nomic-embed-text-v1.5': {
                dimension: 768,
                maxTokens: 8192,
                description: 'Nomic embed text model (8192 token limit)'
            },
            // E5 models
            'intfloat/e5-large-v2': {
                dimension: 1024,
                maxTokens: 512,
                description: 'E5 large English embedding model (512 token limit)'
            },
            'intfloat/e5-base-v2': {
                dimension: 768,
                maxTokens: 512,
                description: 'E5 base English embedding model (512 token limit)'
            }
        };
    }
} 