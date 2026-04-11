package com.phonecluster.app.utils

import android.content.Context
import android.net.Uri
import android.webkit.MimeTypeMap
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import java.io.BufferedReader
import java.io.InputStreamReader
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader

object FileTextExtractor {

    fun extractText(context: Context, uri: Uri): String {

        val mimeType = context.contentResolver.getType(uri)

        return when {
            mimeType?.contains("text") == true -> extractTxt(context, uri)
            mimeType?.contains("pdf") == true -> extractPdf(context, uri)
            else -> ""
        }
    }

    private fun extractTxt(context: Context, uri: Uri): String {
        val inputStream = context.contentResolver.openInputStream(uri)
        val reader = BufferedReader(InputStreamReader(inputStream))
        val text = reader.readText()
        reader.close()
        return text
    }

    private fun extractPdf(context: Context, uri: Uri): String {

        PDFBoxResourceLoader.init(context)

        context.contentResolver.openInputStream(uri).use { inputStream ->
            val document = PDDocument.load(inputStream)
            val stripper = PDFTextStripper()
            val text = stripper.getText(document)
            document.close()
            return text
        }
    }

}
