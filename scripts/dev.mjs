import { spawn } from "node:child_process";

let shuttingDown = false;

const commands = [
    ["api", ["run", "dev:api"]],
    ["web", ["run", "dev:web"]],
];

const children = commands.map(([name, args]) => {
    const child = spawn("npm", args, {
        stdio: "pipe",
        shell: process.platform === "win32",
        detached: process.platform !== "win32",
        env: process.env,
    });

    child.stdout.on("data", (chunk) => write(name, chunk));
    child.stderr.on("data", (chunk) => write(name, chunk));
    child.on("exit", (code, signal) => {
        if (shuttingDown) return;
        console.error(`[${name}] exited with ${signal ?? code}`);
        shutdown(code ?? 1);
    });

    return child;
});

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));
process.on("exit", () => killChildren());

function write(name, chunk) {
    for (const line of String(chunk).split(/\r?\n/)) {
        if (line) console.log(`[${name}] ${line}`);
    }
}

function shutdown(code) {
    shuttingDown = true;
    killChildren();
    process.exitCode = code;
}

function killChildren() {
    for (const child of children) {
        if (child.killed || child.exitCode !== null) continue;
        if (process.platform === "win32") {
            child.kill();
        } else if (child.pid) {
            try {
                process.kill(-child.pid, "SIGTERM");
            } catch {
                child.kill();
            }
        }
    }
}
