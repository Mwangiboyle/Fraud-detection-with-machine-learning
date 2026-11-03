import { getMetadata, predict, ValidationError } from "./lib/inference.mjs";

const MAX_BODY_BYTES = 64 * 1024;

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
    },
  });
}

export default async (request) => {
  if (request.method === "GET") {
    return json(getMetadata());
  }

  if (request.method !== "POST") {
    return json({ error: "Method not allowed." }, 405);
  }

  const contentLength = Number(request.headers.get("content-length") || 0);
  if (contentLength > MAX_BODY_BYTES) {
    return json({ error: "Request body is too large." }, 413);
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return json({ error: "Request body must be valid JSON." }, 400);
  }

  try {
    const result = predict(body?.model, body?.features);
    return json(result);
  } catch (error) {
    if (error instanceof ValidationError) {
      return json({ error: error.message }, 400);
    }
    console.error(error);
    return json({ error: "Unable to run the prediction." }, 500);
  }
};

export const config = {
  path: "/api/predict",
};
