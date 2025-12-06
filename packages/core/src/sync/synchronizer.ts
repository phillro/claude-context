import * as fs from 'fs/promises';
import * as path from 'path';
import * as crypto from 'crypto';
import { MerkleDAG } from './merkle';
import * as os from 'os';

export class FileSynchronizer {
    private fileHashes: Map<string, string>;
    private merkleDAG: MerkleDAG;
    private rootDir: string;
    private snapshotPath: string;
    private ignorePatterns: string[];
    // Pending state for transactional updates - only committed after successful indexing
    private pendingFileHashes?: Map<string, string>;
    private pendingMerkleDAG?: MerkleDAG;

    constructor(rootDir: string, ignorePatterns: string[] = []) {
        this.rootDir = rootDir;
        this.snapshotPath = this.getSnapshotPath(rootDir);
        this.fileHashes = new Map();
        this.merkleDAG = new MerkleDAG();
        this.ignorePatterns = ignorePatterns;
    }

    private getSnapshotPath(codebasePath: string): string {
        const homeDir = os.homedir();
        const merkleDir = path.join(homeDir, '.context', 'merkle');

        const normalizedPath = path.resolve(codebasePath);
        const hash = crypto.createHash('md5').update(normalizedPath).digest('hex');

        return path.join(merkleDir, `${hash}.json`);
    }

    private async hashFile(filePath: string): Promise<string> {
        // Double-check that this is actually a file, not a directory
        const stat = await fs.stat(filePath);
        if (stat.isDirectory()) {
            throw new Error(`Attempted to hash a directory: ${filePath}`);
        }
        const content = await fs.readFile(filePath, 'utf-8');
        return crypto.createHash('sha256').update(content).digest('hex');
    }

    private async generateFileHashes(dir: string): Promise<Map<string, string>> {
        const fileHashes = new Map<string, string>();

        let entries;
        try {
            entries = await fs.readdir(dir, { withFileTypes: true });
        } catch (error: any) {
            console.warn(`Cannot read directory ${dir}: ${error.message}`);
            return fileHashes;
        }

        for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);
            const relativePath = path.relative(this.rootDir, fullPath);

            // Check if this path should be ignored BEFORE any file system operations
            if (this.shouldIgnore(relativePath, entry.isDirectory())) {
                continue; // Skip completely - no access at all
            }

            // Double-check with fs.stat to be absolutely sure about file type
            let stat;
            try {
                stat = await fs.stat(fullPath);
            } catch (error: any) {
                console.warn(`Cannot stat ${fullPath}: ${error.message}`);
                continue;
            }

            if (stat.isDirectory()) {
                // Verify it's really a directory and not ignored
                if (!this.shouldIgnore(relativePath, true)) {
                    const subHashes = await this.generateFileHashes(fullPath);
                    const entries = Array.from(subHashes.entries());
                    for (let i = 0; i < entries.length; i++) {
                        const [p, h] = entries[i];
                        fileHashes.set(p, h);
                    }
                }
            } else if (stat.isFile()) {
                // Verify it's really a file and not ignored
                if (!this.shouldIgnore(relativePath, false)) {
                    try {
                        const hash = await this.hashFile(fullPath);
                        fileHashes.set(relativePath, hash);
                    } catch (error: any) {
                        console.warn(`Cannot hash file ${fullPath}: ${error.message}`);
                        continue;
                    }
                }
            }
            // Skip other types (symlinks, etc.)
        }
        return fileHashes;
    }

    private shouldIgnore(relativePath: string, isDirectory: boolean = false): boolean {
        // Always ignore hidden files and directories (starting with .)
        const pathParts = relativePath.split(path.sep);
        if (pathParts.some(part => part.startsWith('.'))) {
            return true;
        }

        if (this.ignorePatterns.length === 0) {
            return false;
        }

        // Normalize path separators and remove leading/trailing slashes
        const normalizedPath = relativePath.replace(/\\/g, '/').replace(/^\/+|\/+$/g, '');

        if (!normalizedPath) {
            return false; // Don't ignore root
        }

        // Check direct pattern matches first
        for (const pattern of this.ignorePatterns) {
            if (this.matchPattern(normalizedPath, pattern, isDirectory)) {
                return true;
            }
        }

        // Check if any parent directory is ignored
        const normalizedPathParts = normalizedPath.split('/');
        for (let i = 0; i < normalizedPathParts.length; i++) {
            const partialPath = normalizedPathParts.slice(0, i + 1).join('/');
            for (const pattern of this.ignorePatterns) {
                // Check directory patterns
                if (pattern.endsWith('/')) {
                    const dirPattern = pattern.slice(0, -1);
                    if (this.simpleGlobMatch(partialPath, dirPattern) ||
                        this.simpleGlobMatch(normalizedPathParts[i], dirPattern)) {
                        return true;
                    }
                }
                // Check exact path patterns
                else if (pattern.includes('/')) {
                    if (this.simpleGlobMatch(partialPath, pattern)) {
                        return true;
                    }
                }
                // Check filename patterns against any path component
                else {
                    if (this.simpleGlobMatch(normalizedPathParts[i], pattern)) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    private matchPattern(filePath: string, pattern: string, isDirectory: boolean = false): boolean {
        // Clean both path and pattern
        const cleanPath = filePath.replace(/^\/+|\/+$/g, '');
        const cleanPattern = pattern.replace(/^\/+|\/+$/g, '');

        if (!cleanPath || !cleanPattern) {
            return false;
        }

        // Handle directory patterns (ending with /)
        if (pattern.endsWith('/')) {
            if (!isDirectory) return false; // Directory pattern only matches directories
            const dirPattern = cleanPattern.slice(0, -1);

            // Direct match or any path component matches
            return this.simpleGlobMatch(cleanPath, dirPattern) ||
                cleanPath.split('/').some(part => this.simpleGlobMatch(part, dirPattern));
        }

        // Handle path patterns (containing /)
        if (cleanPattern.includes('/')) {
            return this.simpleGlobMatch(cleanPath, cleanPattern);
        }

        // Handle filename patterns (no /) - match against basename
        const fileName = path.basename(cleanPath);
        return this.simpleGlobMatch(fileName, cleanPattern);
    }

    private simpleGlobMatch(text: string, pattern: string): boolean {
        if (!text || !pattern) return false;

        // Convert glob pattern to regex
        const regexPattern = pattern
            .replace(/[.+^${}()|[\]\\]/g, '\\$&') // Escape regex special chars except *
            .replace(/\*/g, '.*'); // Convert * to .*

        const regex = new RegExp(`^${regexPattern}$`);
        return regex.test(text);
    }

    private buildMerkleDAG(fileHashes: Map<string, string>): MerkleDAG {
        const dag = new MerkleDAG();
        const keys = Array.from(fileHashes.keys());
        const sortedPaths = keys.slice().sort(); // Create a sorted copy

        // Create a root node for the entire directory
        let valuesString = "";
        keys.forEach(key => {
            valuesString += fileHashes.get(key);
        });
        const rootNodeData = "root:" + valuesString;
        const rootNodeId = dag.addNode(rootNodeData);

        // Add each file as a child of the root
        for (const path of sortedPaths) {
            const fileData = path + ":" + fileHashes.get(path);
            dag.addNode(fileData, rootNodeId);
        }

        return dag;
    }

    public async initialize() {
        console.log(`Initializing file synchronizer for ${this.rootDir}`);
        await this.loadSnapshot();
        this.merkleDAG = this.buildMerkleDAG(this.fileHashes);
        console.log(`File synchronizer initialized. Loaded ${this.fileHashes.size} file hashes.`);
    }

    public async checkForChanges(): Promise<{ added: string[], removed: string[], modified: string[] }> {
        console.log('Checking for file changes...');

        const newFileHashes = await this.generateFileHashes(this.rootDir);
        const newMerkleDAG = this.buildMerkleDAG(newFileHashes);

        // Compare the DAGs
        const changes = MerkleDAG.compare(this.merkleDAG, newMerkleDAG);

        // If there are any changes in the DAG, we should also do a file-level comparison
        if (changes.added.length > 0 || changes.removed.length > 0 || changes.modified.length > 0) {
            console.log('Merkle DAG has changed. Comparing file states...');
            const fileChanges = this.compareStates(this.fileHashes, newFileHashes);

            // Store pending changes in memory but DON'T save to disk yet
            // The snapshot should only be saved after successful vector indexing
            // via commitChanges() to maintain consistency between merkle state and vectors
            this.pendingFileHashes = newFileHashes;
            this.pendingMerkleDAG = newMerkleDAG;

            console.log(`Found changes: ${fileChanges.added.length} added, ${fileChanges.removed.length} removed, ${fileChanges.modified.length} modified.`);
            return fileChanges;
        }

        console.log('No changes detected based on Merkle DAG comparison.');
        return { added: [], removed: [], modified: [] };
    }

    /**
     * Commit pending changes to the snapshot after successful vector indexing.
     * This ensures the merkle snapshot stays in sync with the actual vector database.
     */
    public async commitChanges(): Promise<void> {
        if (this.pendingFileHashes && this.pendingMerkleDAG) {
            this.fileHashes = this.pendingFileHashes;
            this.merkleDAG = this.pendingMerkleDAG;
            await this.saveSnapshot();
            console.log('✅ Committed changes to merkle snapshot');

            // Clear pending state
            this.pendingFileHashes = undefined;
            this.pendingMerkleDAG = undefined;
        }
    }

    /**
     * Discard pending changes (e.g., if vector indexing failed).
     */
    public discardPendingChanges(): void {
        this.pendingFileHashes = undefined;
        this.pendingMerkleDAG = undefined;
        console.log('⚠️ Discarded pending merkle changes');
    }

    private compareStates(oldHashes: Map<string, string>, newHashes: Map<string, string>): { added: string[], removed: string[], modified: string[] } {
        const added: string[] = [];
        const removed: string[] = [];
        const modified: string[] = [];

        const newEntries = Array.from(newHashes.entries());
        for (let i = 0; i < newEntries.length; i++) {
            const [file, hash] = newEntries[i];
            if (!oldHashes.has(file)) {
                added.push(file);
            } else if (oldHashes.get(file) !== hash) {
                modified.push(file);
            }
        }

        const oldKeys = Array.from(oldHashes.keys());
        for (let i = 0; i < oldKeys.length; i++) {
            const file = oldKeys[i];
            if (!newHashes.has(file)) {
                removed.push(file);
            }
        }

        return { added, removed, modified };
    }

    public getFileHash(filePath: string): string | undefined {
        return this.fileHashes.get(filePath);
    }

    private async saveSnapshot(): Promise<void> {
        const merkleDir = path.dirname(this.snapshotPath);
        await fs.mkdir(merkleDir, { recursive: true });

        // Convert Map to array without using iterator
        const fileHashesArray: [string, string][] = [];
        const keys = Array.from(this.fileHashes.keys());
        keys.forEach(key => {
            fileHashesArray.push([key, this.fileHashes.get(key)!]);
        });

        const data = JSON.stringify({
            fileHashes: fileHashesArray,
            merkleDAG: this.merkleDAG.serialize()
        });
        await fs.writeFile(this.snapshotPath, data, 'utf-8');
        console.log(`Saved snapshot to ${this.snapshotPath}`);
    }

    /**
     * Load snapshot from disk. Returns true if snapshot was found, false if not.
     * When snapshot is not found, starts with empty state so all files appear as "added".
     */
    private async loadSnapshot(): Promise<boolean> {
        try {
            const data = await fs.readFile(this.snapshotPath, 'utf-8');
            const obj = JSON.parse(data);

            // Reconstruct Map without using constructor with iterator
            this.fileHashes = new Map();
            for (const [key, value] of obj.fileHashes) {
                this.fileHashes.set(key, value);
            }

            if (obj.merkleDAG) {
                this.merkleDAG = MerkleDAG.deserialize(obj.merkleDAG);
            }
            console.log(`Loaded snapshot from ${this.snapshotPath}`);
            return true;
        } catch (error: any) {
            if (error.code === 'ENOENT') {
                // DON'T auto-generate snapshot here - start with empty state
                // This ensures checkForChanges() will detect all files as "added"
                // and properly sync them to the vector database
                console.log(`Snapshot file not found at ${this.snapshotPath}. Starting with empty state.`);
                this.fileHashes = new Map();
                this.merkleDAG = new MerkleDAG();
                return false;
            } else {
                throw error;
            }
        }
    }

    /**
     * Delete snapshot file for a given codebase path
     */
    static async deleteSnapshot(codebasePath: string): Promise<void> {
        const homeDir = os.homedir();
        const merkleDir = path.join(homeDir, '.context', 'merkle');
        const normalizedPath = path.resolve(codebasePath);
        const hash = crypto.createHash('md5').update(normalizedPath).digest('hex');
        const snapshotPath = path.join(merkleDir, `${hash}.json`);

        try {
            await fs.unlink(snapshotPath);
            console.log(`Deleted snapshot file: ${snapshotPath}`);
        } catch (error: any) {
            if (error.code === 'ENOENT') {
                console.log(`Snapshot file not found (already deleted): ${snapshotPath}`);
            } else {
                console.error(`Failed to delete snapshot file ${snapshotPath}:`, error.message);
                throw error; // Re-throw non-ENOENT errors
            }
        }
    }
}