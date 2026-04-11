package com.phonecluster.app.ml

import com.phonecluster.app.ml.summary.Text2Summary
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

object SummaryEngine {

    /**
     * Generates an extractive TF-IDF summary.
     *
     * @param text The raw extracted file content.
     * @param compressionRate Value between 0.0 and 1.0.
     *        Example:
     *        0.3 → keep 30% of sentences
     *        0.2 → keep 20% of sentences
     */
    suspend fun summarize(
        text: String,
        compressionRate: Float = 0.3f
    ): String {

        if (text.isBlank()) return ""

        return withContext(Dispatchers.Default) {
            Text2Summary.summarize(text, compressionRate)
        }
    }
}
