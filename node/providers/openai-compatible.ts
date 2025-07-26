import OpenAI from "openai";
import type {
  Provider,
  ProviderMessage,
  ProviderStreamEvent,
  ProviderStreamRequest,
  ProviderToolSpec,
  ProviderToolUseRequest,
  ProviderToolUseResponse,
  StopReason,
  Usage,
} from "./provider-types";
import type { Nvim } from "../nvim/nvim-node";
import { DEFAULT_SYSTEM_PROMPT } from "./system-prompt";
import { assertUnreachable } from "../utils/assertUnreachable";
import type {
  JSONSchemaObject,
  JSONSchemaType,
} from "openai/lib/jsonschema.mjs";
import { validateInput } from "../tools/helpers";
import type { Result } from "../utils/result";
import type { ToolRequest, ToolRequestId } from "../tools/types";
import type { Stream } from "openai/streaming";

export class OpenAICompatibelProvider implements Provider {
  private client: OpenAI;

  constructor(
    _nvim: Nvim,
    options?: {
      baseUrl?: string | undefined;
      apiKeyEnvVar?: string | undefined;
    },
  ) {
    const apiKeyEnvVar = options?.apiKeyEnvVar || "OPENAI_COMP_API_KEY";
    const apiKey = process.env[apiKeyEnvVar];

    if (!apiKey) {
      throw new Error(`${apiKeyEnvVar} not found in environment`);
    }

    this.client = new OpenAI({
      apiKey,
      baseURL: options?.baseUrl || process.env.OPENAI_COMP_BASE_URL,
    });
  }

  createStreamParameters(options: {
    model: string;
    messages: Array<ProviderMessage>;
    tools: Array<ProviderToolSpec>;
    disableCaching?: boolean;
    systemPrompt?: string;
  }): OpenAI.Chat.ChatCompletionCreateParamsStreaming {
    const { messages, tools, systemPrompt } = options;
    const chatMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: systemPrompt || DEFAULT_SYSTEM_PROMPT,
      },
    ];

    for (const message of messages) {
      for (const content of message.content) {
        switch (content.type) {
          case "text":
            chatMessages.push({
              role: message.role,
              content: content.text,
            });
            break;

          case "tool_use": {
            let args: Record<string, unknown>;
            if (content.request.status === "ok") {
              args = content.request.value.input as Record<string, unknown>;
            } else {
              args = content.request.rawRequest as Record<string, unknown>;
            }

            const toolCall: OpenAI.Chat.ChatCompletionMessageToolCall = {
              id: content.id,
              type: "function",
              function: {
                name: content.name,
                arguments: JSON.stringify(args),
              },
            };

            chatMessages.push({
              role: "assistant",
              content: null,
              tool_calls: [toolCall],
            });
            break;
          }

          case "tool_result": {
            const result =
              content.result.status === "ok"
                ? this.formatToolResult(content.result.value)
                : content.result.error;

            chatMessages.push({
              role: "tool",
              tool_call_id: content.id,
              content:
                typeof result === "string" ? result : JSON.stringify(result),
            });
            break;
          }

          case "server_tool_use":
            throw new Error(
              "Server tool use not implemented for Copilot provider",
            );

          case "web_search_tool_result":
            throw new Error(
              "Web search tool result not implemented for Copilot provider",
            );

          case "image":
            if (message.role === "user") {
              chatMessages.push({
                role: "user",
                content: [
                  {
                    type: "image_url",
                    image_url: {
                      url: `data:${content.source.media_type};base64,${content.source.data}`,
                    },
                  },
                ],
              });
            }
            // Images are not supported for assistant messages in chat completions
            break;

          case "document":
            // Documents are not supported in chat completions API
            chatMessages.push({
              role: message.role,
              content: `[Document: ${content.title || "untitled.pdf"}]`,
            });
            break;

          case "thinking":
            // Thinking content is not directly supported, so add it as a note
            chatMessages.push({
              role: "assistant",
              content: `[Thinking: ${content.thinking}]`,
            });
            break;

          case "redacted_thinking":
            // Redacted thinking content is not directly supported, so add it as a note
            chatMessages.push({
              role: "assistant",
              content: `[Redacted Thinking]`,
            });
            break;

          default:
            assertUnreachable(content);
        }
      }
    }

    return {
      model: options.model,
      stream: true,
      messages: chatMessages,
      tools: tools.map((spec) => {
        const compatibleSpec = this.makeOpenAICompatible(spec);
        return {
          type: "function",
          function: {
            name: compatibleSpec.name,
            description: compatibleSpec.description,
            parameters:
              compatibleSpec.input_schema as OpenAI.FunctionParameters,
          },
        };
      }),
    };
  }

  private sanitizeSchemaForOpenAI(schema: JSONSchemaType): JSONSchemaType {
    if (
      typeof schema !== "object" ||
      schema === null ||
      Array.isArray(schema)
    ) {
      return schema;
    }

    const sanitized = { ...schema };

    // Remove unsupported format specifiers
    if ("format" in sanitized && sanitized.format) {
      const unsupportedFormats = [
        "uri",
        "uri-reference",
        "uri-template",
        "date-time",
        "date",
        "time",
        "email",
        "hostname",
        "ipv4",
        "ipv6",
        "uuid",
        "regex",
        "json-pointer",
      ];

      if (unsupportedFormats.includes(sanitized.format as string)) {
        delete sanitized.format;
        // Add a description hint if not already present
        if (!("description" in sanitized) || !sanitized.description) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
          switch ((schema as any).format) {
            case "uri":
            case "uri-reference":
              sanitized.description = "A valid URI string";
              break;
            case "date-time":
              sanitized.description =
                'A date-time string (e.g., "2023-12-01T10:30:00Z")';
              break;
            case "date":
              sanitized.description = 'A date string (e.g., "2023-12-01")';
              break;
            case "email":
              sanitized.description = "A valid email address";
              break;
            default:
              sanitized.description = `A string in ${JSON.stringify(schema.format)} format`;
          }
        }
      }
    }

    // Recursively sanitize nested objects
    for (const [key, value] of Object.entries(sanitized)) {
      if (key !== "format" && typeof value === "object" && value !== null) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
        (sanitized as any)[key] = this.sanitizeSchemaForOpenAI(value);
      }
    }

    return sanitized;
  }
  private makeOpenAICompatible(spec: ProviderToolSpec): ProviderToolSpec {
    const schema = spec.input_schema;

    // First apply format sanitization
    const sanitizedSchema = this.sanitizeSchemaForOpenAI(schema);

    // Then apply OpenAI-specific requirements
    if (
      typeof sanitizedSchema !== "object" ||
      sanitizedSchema === null ||
      Array.isArray(sanitizedSchema) ||
      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (sanitizedSchema as any).type !== "object"
    ) {
      return { ...spec, input_schema: sanitizedSchema };
    }

    // copy to avoid mutating the underlying spec
    const compatibleSchema = JSON.parse(
      JSON.stringify(sanitizedSchema),
    ) as JSONSchemaObject;

    compatibleSchema.additionalProperties = false;

    if (
      compatibleSchema.properties &&
      typeof compatibleSchema.properties === "object"
    ) {
      const propertyNames = Object.keys(compatibleSchema.properties);
      compatibleSchema.required = propertyNames;
    } else {
      compatibleSchema.required = [];
    }

    return {
      ...spec,
      input_schema: compatibleSchema,
    };
  }

  private formatToolResult(
    value: Array<{ type: string; text?: string; [key: string]: unknown }>,
  ): string {
    return value
      .map((item) => {
        if (item.type === "text" && typeof item.text === "string") {
          return item.text;
        }
        return JSON.stringify(item);
      })
      .join("\n");
  }

  forceToolUse(options: {
    model: string;
    messages: Array<ProviderMessage>;
    spec: ProviderToolSpec;
    systemPrompt?: string;
  }): ProviderToolUseRequest {
    const { messages, spec, systemPrompt } = options;
    let aborted = false;
    const promise = (async (): Promise<ProviderToolUseResponse> => {
      const client = this.client;

      const chatMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
        {
          role: "system",
          content: systemPrompt || DEFAULT_SYSTEM_PROMPT,
        },
      ];

      // Simple message conversion for forced tool use (text only)
      for (const message of messages) {
        for (const content of message.content) {
          if (content.type === "text") {
            chatMessages.push({
              role: message.role,
              content: content.text,
            });
          }
        }
      }

      const response = await client.chat.completions.create({
        model: options.model,
        messages: chatMessages,
        tools: [
          {
            type: "function",
            function: {
              name: spec.name,
              description: spec.description,
              parameters: this.makeOpenAICompatible(spec)
                .input_schema as OpenAI.FunctionParameters,
            },
          },
        ],
        tool_choice: { type: "function", function: { name: spec.name } },
        stream: false,
      });

      // Extract and validate tool call
      const toolCall = response.choices[0]?.message?.tool_calls?.[0];
      if (!toolCall) {
        throw new Error("No tool call in forced response");
      }

      const input = validateInput(
        spec.name,
        JSON.parse(toolCall.function.arguments) as Record<string, unknown>,
      );

      const toolRequest: Result<ToolRequest, { rawRequest: unknown }> =
        input.status === "ok"
          ? {
              status: "ok",
              value: {
                toolName: spec.name,
                id: toolCall.id as ToolRequestId,
                input: input.value,
              },
            }
          : { ...input, rawRequest: toolCall.function.arguments };

      const usage: Usage = response.usage
        ? {
            inputTokens: response.usage.prompt_tokens,
            outputTokens: response.usage.completion_tokens,
          }
        : {
            inputTokens: 0,
            outputTokens: 0,
          };

      if (aborted) {
        throw new Error("Aborted");
      }

      return {
        toolRequest,
        stopReason: "tool_use" as const,
        usage,
      };
    })();

    return {
      abort: () => {
        aborted = true;
      },
      promise,
    };
  }

  sendMessage(options: {
    model: string;
    messages: Array<ProviderMessage>;
    onStreamEvent: (event: ProviderStreamEvent) => void;
    tools: Array<ProviderToolSpec>;
    systemPrompt?: string;
    thinking?: {
      enabled: boolean;
      budgetTokens?: number;
    };
  }): ProviderStreamRequest {
    const { model, messages, onStreamEvent, tools, systemPrompt } = options;
    let streamRequest: Stream<OpenAI.Chat.ChatCompletionChunk>;
    let currentContentBlockIndex = 0;
    let blockStarted = false;

    // Track accumulated tool call data across chunks
    const toolCallAccumulator = new Map<
      number,
      {
        id: string;
        name: string;
        arguments: string;
        blockIndex?: number;
      }
    >();

    const promise = (async (): Promise<{
      usage: Usage;
      stopReason: StopReason;
    }> => {
      const client = this.client;

      streamRequest = await client.chat.completions.create(
        this.createStreamParameters({
          model,
          messages,
          tools,
          ...(systemPrompt && { systemPrompt }),
        }),
      );

      // Start first content block
      onStreamEvent({
        type: "content_block_start",
        index: currentContentBlockIndex,
        content_block: {
          type: "text",
          text: "",
          citations: null,
        },
      });
      blockStarted = true;

      let stopReason: StopReason = "end_turn";
      let usage: Usage = { inputTokens: 0, outputTokens: 0 };

      for await (const chunk of streamRequest) {
        const delta = chunk.choices[0]?.delta;

        if (delta?.content) {
          // Text content delta
          onStreamEvent({
            type: "content_block_delta",
            index: currentContentBlockIndex,
            delta: {
              type: "text_delta",
              text: delta.content,
            },
          });
        }

        if (delta?.tool_calls) {
          // Accumulate tool call data across chunks
          for (const toolCallDelta of delta.tool_calls) {
            const toolCallIndex = toolCallDelta.index || 0;

            if (!toolCallAccumulator.has(toolCallIndex)) {
              toolCallAccumulator.set(toolCallIndex, {
                id: "",
                name: "",
                arguments: "",
              });
            }

            const accumulated = toolCallAccumulator.get(toolCallIndex)!;

            // Accumulate data
            if (toolCallDelta.id) {
              accumulated.id = toolCallDelta.id;
            }
            if (toolCallDelta.function?.name) {
              accumulated.name = toolCallDelta.function.name;
            }
            if (toolCallDelta.function?.arguments) {
              accumulated.arguments += toolCallDelta.function.arguments;
            }

            // Only create content block when we have complete tool info
            if (
              accumulated.id &&
              accumulated.name &&
              accumulated.blockIndex === undefined
            ) {
              // Close any existing text block
              if (blockStarted) {
                onStreamEvent({
                  type: "content_block_stop",
                  index: currentContentBlockIndex,
                });
                blockStarted = false;
              }

              // Start new tool use block
              currentContentBlockIndex++;
              accumulated.blockIndex = currentContentBlockIndex;
              stopReason = "tool_use";

              onStreamEvent({
                type: "content_block_start",
                index: currentContentBlockIndex,
                content_block: {
                  type: "tool_use",
                  id: accumulated.id,
                  name: accumulated.name,
                  input: {},
                },
              });
            }

            // Send argument deltas if we have a block started
            if (
              accumulated.blockIndex !== undefined &&
              toolCallDelta.function?.arguments
            ) {
              onStreamEvent({
                type: "content_block_delta",
                index: accumulated.blockIndex,
                delta: {
                  type: "input_json_delta",
                  partial_json: toolCallDelta.function.arguments,
                },
              });
            }
          }
        }

        // Extract usage if available
        if (chunk.usage) {
          usage = {
            inputTokens: chunk.usage.prompt_tokens || 0,
            outputTokens: chunk.usage.completion_tokens || 0,
          };
        }
      }

      // Final cleanup - close any open blocks
      if (blockStarted) {
        onStreamEvent({
          type: "content_block_stop",
          index: currentContentBlockIndex,
        });
      }

      // Close any tool use blocks
      for (const accumulated of toolCallAccumulator.values()) {
        if (accumulated.blockIndex !== undefined) {
          onStreamEvent({
            type: "content_block_stop",
            index: accumulated.blockIndex,
          });
        }
      }

      return {
        stopReason,
        usage,
      };
    })();

    return {
      abort: () => streamRequest?.controller?.abort(),
      promise,
    };
  }
}
