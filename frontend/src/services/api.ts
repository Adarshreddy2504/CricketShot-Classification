// API Service for connecting to the FastAPI Backend

export interface InferenceResult {
  predictedClass: string;
  confidence: number;
  timeline: { frame: number; confidence: number }[];
  class_probabilities?: Record<string, number>;
}

export const api = {
  predictShot: async (file: File): Promise<InferenceResult> => {
    const formData = new FormData();
    formData.append("video", file);

    const response = await fetch("http://localhost:8000/api/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    const data = await response.json();
    
    return {
      predictedClass: data.prediction,
      confidence: data.confidence * 100, // Convert to percentage
      timeline: data.timeline,
      class_probabilities: data.class_probabilities
    };
  }
};
