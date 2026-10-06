/**
 * Optional model router: on the first prompt of a session, the local model rates how demanding the
 * request is, and the session switches to the user's role for that tier: light -> `@smol`,
 * standard -> `@mid` (or `@default` when no `mid` role is set), complex -> `@slow`. The plugin
 * assigns no models; a tier whose roles are unset keeps the current model.
 */
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { ENDPOINT, serverReady } from "./server";

/** Tier label -> [description shown to the classifier, roles to try in order]. Highest probability wins. */
const TIERS: Record<string, [string, string[]]> = {
	light: ["Answering a question about the code, or a tiny mechanical edit such as a typo, rename, or version bump", ["@smol"]],
	standard: ["Ordinary features, bug fixes, tests, or code reviews within one area", ["@mid", "@default"]],
	complex: ["Subtle design, cross-cutting changes, or hard debugging", ["@slow"]],
};

export default function (pi: ExtensionAPI) {
	let routed = false;
	pi.on("session_start", async () => {
		routed = false;
	});
	pi.on("before_agent_start", async (event, ctx) => {
		if (routed) return;
		routed = true;
		await serverReady();
		let tier: string;
		let probabilities: Record<string, number>;
		try {
			const response = await fetch(ENDPOINT, {
				method: "POST",
				headers: { "content-type": "application/json" },
				body: JSON.stringify({
					state: { prompt: event.prompt.slice(0, 16_000) },
					questions: {
						tier: {
							type: "choice",
							instructions: "How demanding is the software engineering work requested in `prompt`?",
							criteria: Object.fromEntries(Object.entries(TIERS).map(([label, [description]]) => [label, description])),
						},
					},
				}),
				signal: AbortSignal.timeout(10_000),
			});
			const answer = (await response.json()).answers.tier;
			tier = answer.choice;
			probabilities = answer.probabilities;
		} catch (error) {
			ctx.ui.notify(`bonsai-router: local server unavailable, keeping ${ctx.model?.id} (${error})`, "warning");
			return;
		}
		const roles = TIERS[tier]?.[1] ?? [];
		const role = roles.find((candidate) => ctx.models.resolve(candidate));
		const model = role ? ctx.models.resolve(role) : undefined;
		const ok = model ? await pi.setModel(model) : false;
		const shown = Object.entries(probabilities).map(([label, p]) => `${label}=${p.toFixed(2)}`).join(" ");
		const target = model ? `${role} = ${model.provider}/${model.id}` : `${roles.join("/") || tier} not set, keeping ${ctx.model?.id}`;
		ctx.ui.notify(`bonsai-router: ${shown} -> ${target}${model && !ok ? " (switch failed)" : ""}`, ok ? "info" : "warning");
	});
}
