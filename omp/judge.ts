/**
 * Registers the local server as the System One judgment model `bonsai-local/bonsai-4b-system-one`.
 * Assign it to the `judge` role (`modelRoles.judge`) and omp's judge features, such as the per-prompt
 * `auto` thinking level, use the local model. Starts the server on the first prompt when needed.
 */
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { LOG, SERVER, serverReady } from "./server";

export default function (pi: ExtensionAPI) {
	pi.registerProvider("bonsai-local", {
		baseUrl: SERVER,
		apiKey: "local",
		api: "typesafe",
		models: [
			{
				id: "bonsai-4b-system-one",
				name: "Bonsai 4B System One (local)",
				reasoning: false,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 32_768,
				maxTokens: 0,
			},
		],
	});
	let warned = false;
	// Runs before omp's own judge calls for the turn, so the first prompt waits for a starting server.
	pi.on("before_agent_start", async (_event, ctx) => {
		if (!(await serverReady()) && !warned) {
			warned = true;
			ctx.ui.notify(`bonsai-system-one: local server at ${SERVER} is not answering (log: ${LOG})`, "warning");
		}
	});
}
