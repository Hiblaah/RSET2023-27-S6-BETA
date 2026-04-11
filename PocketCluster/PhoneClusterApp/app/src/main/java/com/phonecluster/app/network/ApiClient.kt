package com.phonecluster.app.network

import com.phonecluster.app.core.SERVER_BASE_URL
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

// singleton instead of a class
// because class will have multiple connection pools (wasteful resource usage)
object ApiClient {
//    private const val BASE_URL = "http://10.124.156.168:8000" //hardcoding this for now WILL CHANGE ACCORDING TO THE
    private const val BASE_URL = SERVER_BASE_URL
    // catches all the HTTP request and response and logs
    //url, header, request (JSON), response (JSON)
    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }

    private val okHttpClient = OkHttpClient.Builder()
        //automatically logged cuz of the interceptor
        .addInterceptor(loggingInterceptor) // all requests flow thru okhttp
        .build()

    //decides where to send the requests, how they are sent, and how responses are decoded
    private val retrofit: Retrofit = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .client(okHttpClient)
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    val apiService: ApiService = retrofit.create(ApiService::class.java)
}