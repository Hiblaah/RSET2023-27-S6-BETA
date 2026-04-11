package com.phonecluster.app.ml

import kotlin.math.sqrt
import kotlin.math.min
object SimilarityUtils {

    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        var normA = 0f
        var normB = 0f

        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        val rawScore = dot / (sqrt(normA) * sqrt(normB) + 1e-10f)

        val safeScore = kotlin.math.max(0f, rawScore)

        val boostedScore = sqrt(safeScore) * 1.3f

        return min(1.00f, boostedScore)
    }
}
