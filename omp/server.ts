/**
 * Shared by the plugin's extensions: the local server address, and a once-per-process start of the
 * server when it is down and BONSAI_SYSTEM_ONE_SERVE_CMD is set.
 */
import { spawn } from "node:child_process";
import { closeSync, mkdirSync, openSync, statSync, unlinkSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export const SERVER = process.env.BONSAI_SYSTEM_ONE_URL ?? "http://127.0.0.1:8765";
export const ENDPOINT = `${SERVER}/v1/systemone`;
export const LOG = join(homedir(), ".cache", "bonsai-system-one", "serve.log");

/** Shell command that starts the server, set per machine (e.g. `<venv>/bin/python -m system_one_bonsai.serve`). */
const SERVE_CMD = process.env.BONSAI_SYSTEM_ONE_SERVE_CMD;
const LOCK = join(homedir(), ".cache", "bonsai-system-one", "serve.lock");
/** Model loading takes a few seconds; the first prompt waits at most this long. */
const READY_TIMEOUT_MS = 20_000;
/** A lock older than this belongs to a launch that died before the server came up. */
const STALE_LOCK_MS = 60_000;

async function healthy(): Promise<boolean> {
	try {
		return (await fetch(`${SERVER}/health`, { signal: AbortSignal.timeout(1_000) })).ok;
	} catch {
		return false;
	}
}

let readiness: Promise<boolean> | undefined;

/**
 * Resolves true once the server answers. Starts it detached (it outlives omp) when it is down and a
 * start command is configured; another omp process holding the lock is left to start it. All
 * extensions in this process share one start.
 */
export function serverReady(): Promise<boolean> {
	readiness ??= (async () => {
		if (await healthy()) return true;
		if (!SERVE_CMD) return false;
		mkdirSync(join(homedir(), ".cache", "bonsai-system-one"), { recursive: true });
		try {
			if (Date.now() - statSync(LOCK).mtimeMs > STALE_LOCK_MS) unlinkSync(LOCK);
		} catch {}
		let owner = false;
		try {
			closeSync(openSync(LOCK, "wx"));
			owner = true;
			const log = openSync(LOG, "a");
			spawn("/bin/sh", ["-c", `exec ${SERVE_CMD}`], { detached: true, stdio: ["ignore", log, log] }).unref();
			closeSync(log);
		} catch {}
		const deadline = Date.now() + READY_TIMEOUT_MS;
		let up = false;
		while (!up && Date.now() < deadline) {
			await Bun.sleep(500);
			up = await healthy();
		}
		if (owner) {
			try {
				unlinkSync(LOCK);
			} catch {}
		}
		return up;
	})();
	return readiness;
}
