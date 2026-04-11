package com.phonecluster.app.screens

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.phonecluster.app.storage.AppDatabase
import com.phonecluster.app.storage.FileEntity
import com.phonecluster.app.utils.FileRepository
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class FileBrowserViewModel(application: Application) : AndroidViewModel(application) {

    private val dao = AppDatabase.getDatabase(application).fileDao()
    private val repository = FileRepository()

    val files: StateFlow<List<FileEntity>> =
        dao.getAllFiles()
            .stateIn(
                viewModelScope,
                SharingStarted.WhileSubscribed(5000),
                emptyList()
            )

    fun downloadFile(fileId: Long) {

        viewModelScope.launch {

            try {

                val file = dao.getFileByServerId(fileId)

                if (file != null) {

                    repository.downloadFile(
                        context = getApplication(),
                        fileId = fileId,
                        fileName = file.fileName
                    )

                } else {
                    Log.e("DOWNLOAD", "File metadata not found for id $fileId")
                }

            } catch (e: Exception) {
                Log.e("DOWNLOAD", "Download failed: ${e.message}")
            }
        }
    }

    fun deleteFile(fileId: Long) {

        viewModelScope.launch {

            // Optimistic local delete
            dao.deleteFileByServerId(fileId)

            try {
                repository.deleteFile(fileId)
            } catch (e: Exception) {
                Log.e("DELETE", "Remote delete failed: ${e.message}")
            }
        }
    }
}