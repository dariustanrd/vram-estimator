import app from "../../apps/api/src/index";

type Env = {
    HF_TOKEN?: string;
};

type PagesFunctionContext = {
    request: Request;
    env: Env;
    waitUntil: (promise: Promise<unknown>) => void;
    passThroughOnException: () => void;
};

export const onRequest = (
    context: PagesFunctionContext,
): Response | Promise<Response> => {
    return app.fetch(
        context.request,
        context.env,
        context as unknown as ExecutionContext,
    );
};
