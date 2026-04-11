package com.phonecluster.app.ml

import android.content.Context
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import kotlin.math.min

class OnnxTokenizer(context: Context) {

    private val vocab: Map<String, Int>

    private val maxLength = 128
    private val unkId = 100
    private val clsId = 101
    private val sepId = 102
    private val padId = 0

    init {
        vocab = loadVocab(context)
    }

    private fun loadVocab(context: Context): Map<String, Int> {
        val inputStream = context.assets.open("tokenizer.json")
        val jsonText = BufferedReader(InputStreamReader(inputStream))
            .use { it.readText() }

        val json = JSONObject(jsonText)
        val model = json.getJSONObject("model")
        val vocabJson = model.getJSONObject("vocab")

        val map = mutableMapOf<String, Int>()

        val keys = vocabJson.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            map[key] = vocabJson.getInt(key)
        }

        return map
    }

    fun tokenize(text: String): Triple<Array<LongArray>, Array<LongArray>, Array<LongArray>> {

        val cleaned = text.lowercase()

        val words = cleaned.split(Regex("\\s+"))

        val wordPieceTokens = mutableListOf<Int>()

        for (word in words) {
            if (word.isBlank()) continue

            val tokens = wordPieceTokenize(word)
            wordPieceTokens.addAll(tokens)
        }

        // Add special tokens
        val inputIds = mutableListOf<Int>()
        inputIds.add(clsId)
        inputIds.addAll(wordPieceTokens)
        inputIds.add(sepId)

        // Truncate if needed
        val truncated = inputIds.take(maxLength)

        // Attention mask (1 for real tokens)
        val attentionMask = MutableList(truncated.size) { 1 }

        // Padding
        while (truncated.size < maxLength) {
            (truncated as MutableList).add(padId)
            attentionMask.add(0)
        }

        val tokenTypeIds = MutableList(maxLength) { 0L }

        return Triple(
            arrayOf(truncated.map { it.toLong() }.toLongArray()),
            arrayOf(attentionMask.map { it.toLong() }.toLongArray()),
            arrayOf(tokenTypeIds.toLongArray())
        )
    }

    private fun wordPieceTokenize(word: String): List<Int> {

        if (vocab.containsKey(word)) {
            return listOf(vocab[word]!!)
        }

        val subTokens = mutableListOf<Int>()
        var start = 0

        while (start < word.length) {
            var end = word.length
            var found = false
            var currentSub: String? = null

            while (start < end) {
                var substr = word.substring(start, end)
                if (start > 0) {
                    substr = "##$substr"
                }

                if (vocab.containsKey(substr)) {
                    currentSub = substr
                    found = true
                    break
                }

                end--
            }

            if (!found) {
                return listOf(unkId)
            }

            subTokens.add(vocab[currentSub]!!)
            start = end
        }

        return subTokens
    }
}
