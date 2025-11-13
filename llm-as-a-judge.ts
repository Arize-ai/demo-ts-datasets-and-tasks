import { openai } from "@ai-sdk/openai";
import { register } from "@arizeai/phoenix-otel";
import {
  createOrGetDataset,
  getDataset,
} from "@arizeai/phoenix-client/datasets";
import { runExperiment } from "@arizeai/phoenix-client/experiments";
import type { ExperimentTask } from "@arizeai/phoenix-client/types/experiments";
import z from "zod";
import { generateText } from "ai";

const { PHOENIX_PROJECT_NAME, PHOENIX_HOST } = process.env;

// Fetch the existing dataset
const dataset = await getDataset({
  dataset: {
    // datasetName: "phoenix-evaluation-dataset-task-quickstart-2025-11-13T18:07:53.237Z", // 4 good examples
    datasetName: "phoenix-evaluation-dataset-task-quickstart-2025-11-13T19:33:51.018Z" // 4 good, 1 bad example
  },
});

// Create a task to judge the dataset examples

const task: ExperimentTask = async (example) => {
  const { question } = z.object({ question: z.string() }).parse(example.input);
  const { answer: groundTruthAnswer } = z.object({ answer: z.string() }).parse(example.input);
  const { answer: llmAnswer } = z.object({ answer: z.string() }).parse(example.output);

  const { text: judgement } = await generateText({
    model: openai("gpt-4o-mini"),
    system: "You are judging whether an LLM's answer to a question is good or not. Respond with 'good' or 'bad'.",
    prompt: `Question: ${question}\nGround truth answer: ${groundTruthAnswer}\nLLM answer: ${llmAnswer}`,
    experimental_telemetry: {
      isEnabled: true,
    },
  });
  return judgement;
};

// Create an experiment that will use the task to answer questions about the dataset

const experiment = await runExperiment({
  experimentName: "phoenix-experiment-task-quickstart",
  experimentDescription: "An experiment for the phoenix task quickstart",
  dataset: { datasetId: dataset.id },
  task,
});

// Apply the task outputs into a new dataset for inspection
console.log(`🔄 Applying task outputs into a new dataset for inspection...`);

const timestamp = new Date().toISOString();
const { datasetId: evaluationDatasetId } = await createOrGetDataset({
  name: `llm-as-a-judge-${timestamp}`,
  description: "A dataset for the llm as a judge task",
  examples: Object.values(experiment.runs).flatMap((run) => {
    const originalExample = dataset.examples.find(
      (example) => example.id === run.datasetExampleId
    );
    if (!originalExample) {
      return [];
    }
    return [
      {
        input: { 
            question: originalExample.input.question,
            groundTruthAnswer: originalExample.input.answer,
            llmAnswer: run.output,
        },
        output: { judgement: run.output },
      },
    ];
  }),
});
