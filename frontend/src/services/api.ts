// API Service for connecting to the FastAPI Backend

export interface InferenceResult {
  predictedClass: string;
  confidence: number;
  timeline: {
    frame: number;
    confidence: number;
    prediction?: string;
  }[];
  class_probabilities?: Record<string, number>;
  clips_processed?: number;
}

export interface TaskResponse {
  task_id: string;
}

export interface ProgressUpdate {
  progress: number;
  status: string;
  cancelled: boolean;
  result?: BackendInferenceResult;
  error?: string;
}

interface BackendInferenceResult {
  prediction: string;
  confidence: number;
  timeline?: {
    frame: number;
    confidence: number;
    prediction?: string;
  }[];
  class_probabilities?: Record<string, number>;
  clips_processed?: number;
}

const API_BASE_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

export const api = {
  /**
   * Upload a video and start an inference task.
   */
  predictShot: async (file: File): Promise<TaskResponse> => {
    const formData = new FormData();
    formData.append("video", file);

    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let message = `API error: ${response.status} ${response.statusText}`;

      try {
        const errorData = await response.json();

        if (errorData?.detail) {
          message = errorData.detail;
        }
      } catch {
        // Keep default error message
      }

      throw new Error(message);
    }

    return await response.json();
  },

  /**
   * Cancel a running inference task.
   */
  cancelTask: async (taskId: string): Promise<void> => {
    const response = await fetch(
      `${API_BASE_URL}/api/cancel/${taskId}`,
      {
        method: "DELETE",
      }
    );

    if (!response.ok) {
      let message = `Cancel error: ${response.status} ${response.statusText}`;

      try {
        const errorData = await response.json();

        if (errorData?.detail) {
          message = errorData.detail;
        }
      } catch {
        // Keep default error message
      }

      throw new Error(message);
    }
  },

  /**
   * Listen to backend Server-Sent Events for task progress.
   */
  streamProgress: (
    taskId: string,
    onProgress: (update: ProgressUpdate) => void,
    onComplete: (result: InferenceResult) => void,
    onError: (error: Error) => void
  ): EventSource => {
    const eventSource = new EventSource(
      `${API_BASE_URL}/api/progress/${taskId}`
    );

    eventSource.onmessage = (event) => {
      try {
        const update: ProgressUpdate = JSON.parse(event.data);

        onProgress(update);

        // Task was cancelled
        if (update.cancelled) {
          eventSource.close();
          return;
        }

        // Task completed
        if (update.result) {
          const backendResult = update.result;

          const result: InferenceResult = {
            predictedClass: backendResult.prediction,

            confidence:
              backendResult.confidence <= 1
                ? backendResult.confidence * 100
                : backendResult.confidence,

            timeline: (backendResult.timeline || []).map((point) => ({
              frame: point.frame,

              // Backend currently sends timeline confidence
              // already as percentage.
              confidence: point.confidence,

              prediction: point.prediction,
            })),

            class_probabilities:
              backendResult.class_probabilities,

            clips_processed:
              backendResult.clips_processed,
          };

          eventSource.close();
          onComplete(result);
        }
      } catch (error) {
        eventSource.close();

        onError(
          error instanceof Error
            ? error
            : new Error("Failed to process progress update")
        );
      }
    };

    eventSource.onerror = () => {
      eventSource.close();

      onError(
        new Error(
          "Lost connection to the inference progress stream."
        )
      );
    };

    return eventSource;
  },
};