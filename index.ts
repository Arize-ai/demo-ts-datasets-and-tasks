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

// Set up Phoenix tracing

register({
  projectName: PHOENIX_PROJECT_NAME,
  // ensure traces are flushed immediately
  batch: false,
});

// Create or get the dataset we will use for this quickstart

const { datasetId } = await createOrGetDataset({
  name: "phoenix-dataset-task-quickstart",
  description: "A dataset for the phoenix dataset task quickstart",
  examples: [
    {
      input: {
        question: "What JS function is commonly used to make http requests?",
      },
      output: { answer: "fetch" },
    },
    {
      input: {
        question:
          "What JS function is commonly used to apply transformation to each element of an array?",
      },
      output: { answer: "map" },
    },
    {
      input: {
        question:
          "What JS data structure is commonly used to store a collection of key-value pairs?",
      },
      output: { answer: "object" },
    },
    {
      input: {
        question:
          "What JS data structure is commonly used to ensure a value is only present once in a collection?",
      },
      output: { answer: "set" },
    },
  ],
});
const dataset = await getDataset({ dataset: { datasetId } });

console.log(`📝 Reference dataset ${dataset.name} (${dataset.id}) created`);
console.log(
  `🔗 You can view this dataset at ${PHOENIX_HOST}/datasets/${dataset.id}/examples`
);

// Create a task that can answer questions about the dataset

const task: ExperimentTask = async (example) => {
  const { question } = z.object({ question: z.string() }).parse(example.input);
  const { text } = await generateText({
    model: openai("gpt-4o-mini"),
    system: "You answer questions as tersely as possible.",
    prompt: question,
    experimental_telemetry: {
      isEnabled: true,
    },
  });
  return text;
};

// Create an experiment that will use the task to answer questions about the dataset

const experiment = await runExperiment({
  experimentName: "phoenix-experiment-task-quickstart",
  experimentDescription: "An experiment for the phoenix task quickstart",
  dataset: { datasetId },
  task,
});

// Apply the task outputs into a new dataset that can be used for evaluation

console.log(`🔄 Applying task outputs into a new dataset for evaluation...`);

const timestamp = new Date().toISOString();
const { datasetId: evaluationDatasetId } = await createOrGetDataset({
  name: `phoenix-evaluation-dataset-task-quickstart-${timestamp}`,
  description: "A dataset for the phoenix task quickstart evaluation",
  examples: Object.values(experiment.runs).flatMap((run) => {
    const originalExample = dataset.examples.find(
      (example) => example.id === run.datasetExampleId
    );
    if (!originalExample) {
      return [];
    }
    return [
      {
        input: { ...originalExample.input, ...originalExample.output },
        output: { answer: run.output },
      },
    ];
  }),
});

const evaluationDataset = await getDataset({
  dataset: { datasetId: evaluationDatasetId },
});
console.log(
  `📝 Evaluation dataset ${evaluationDataset.name} (${evaluationDataset.id}) created`
);
console.log(
  `🔗 You can view this dataset at ${PHOENIX_HOST}/datasets/${evaluationDataset.id}/examples`
);
