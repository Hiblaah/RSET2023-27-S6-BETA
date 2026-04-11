package com.phonecluster.app.utils

import android.content.Context
import android.net.Uri
import java.io.File

object FileHelper {

    fun readAllBytes(context: Context, uri: Uri): ByteArray {
        context.contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Unable to open file" }
            return input.readBytes()
        }
    }

    fun getOutputDir(context: Context): File {
        val dir = File("/storage/emulated/0/Download/ascon_output")
        if (!dir.exists()) {
            dir.mkdirs()
        }
        return dir
    }
}