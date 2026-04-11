package com.phonecluster.app.ml
import android.content.Context
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import android.util.Log

class EmbeddingEngine(private val context: Context) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = loadModelFromAssets(context, "model.onnx")
        val sessionOptions = OrtSession.SessionOptions()
        session = env.createSession(modelBytes, sessionOptions)
        Log.d("ONNX_DEBUG", "Model session created successfully")
    }

    private fun loadModelFromAssets(context: Context, fileName: String): ByteArray {
        return context.assets.open(fileName).readBytes()
    }

    fun generateEmbedding(
        inputIds: Array<LongArray>,
        attentionMask: Array<LongArray>,
        tokenTypeIds: Array<LongArray>
    ): FloatArray {

        val inputIdsTensor = OnnxTensor.createTensor(env, inputIds)
        val attentionMaskTensor = OnnxTensor.createTensor(env, attentionMask)
        val tokenTypeTensor = OnnxTensor.createTensor(env, tokenTypeIds)

        val inputs = mapOf(
            "input_ids" to inputIdsTensor,
            "attention_mask" to attentionMaskTensor,
            "token_type_ids" to tokenTypeTensor
        )

        val inferenceStart = System.currentTimeMillis()
        val results = session.run(inputs)
        val inferenceEnd = System.currentTimeMillis()
        Log.d("PIPELINE_TIMING", "  ONNX session.run() inference: ${inferenceEnd - inferenceStart} ms")

        val output = results[0].value as Array<Array<FloatArray>>

        val poolingStart = System.currentTimeMillis()
        val embedding = meanPooling(output[0], attentionMask[0])
        val poolingEnd = System.currentTimeMillis()
        Log.d("PIPELINE_TIMING", "  Mean pooling: ${poolingEnd - poolingStart} ms")

        return embedding
    }

    private fun meanPooling(
        tokenEmbeddings: Array<FloatArray>,
        mask: LongArray
    ): FloatArray {
        val hiddenSize = tokenEmbeddings[0].size
        val sentenceEmbedding = FloatArray(hiddenSize)
        var validTokens = 0f

        for (i in tokenEmbeddings.indices) {
            if (mask[i] == 1L) {
                validTokens++
                for (j in 0 until hiddenSize) {
                    sentenceEmbedding[j] += tokenEmbeddings[i][j]
                }
            }
        }

        for (j in 0 until hiddenSize) {
            sentenceEmbedding[j] /= validTokens
        }

        return sentenceEmbedding
    }
}
