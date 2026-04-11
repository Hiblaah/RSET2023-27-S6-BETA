document.addEventListener("DOMContentLoaded", function(){

let monitoring = true;

const statusBox = document.getElementById("statusBox");
const monitorBtn = document.getElementById("monitorBtn");

function toggleMonitoring(){

monitoring = !monitoring;

if(monitoring){

statusBox.innerText = "ONLINE";
statusBox.classList.remove("offline");
statusBox.classList.add("online");

monitorBtn.innerText = "Stop Monitoring";

}
else{

statusBox.innerText = "OFFLINE";
statusBox.classList.remove("online");
statusBox.classList.add("offline");

monitorBtn.innerText = "Start Monitoring";

}

}

window.toggleMonitoring = toggleMonitoring;

function loadLogs(){

fetch("/logs")
.then(res => res.json())
.then(data => {

const table = document.getElementById("logTable");


if(!table){
console.log("logTable not found");
return;
}

table.innerHTML = "";

data.forEach(log => {

const row = document.createElement("tr");

row.innerHTML = `
<td>${log.application}</td>
<td>${log.date}</td>
<td>${log.time}</td>
<td>${log.duration}</td>
<td>${log.status}</td>
`;

table.appendChild(row);

});

})
.catch(err => console.log("Error loading logs:", err));

}

function loadAlerts(){

fetch("/alerts")
.then(res => res.json())
.then(data => {

const alertsList = document.getElementById("alertsList");

if(!alertsList) return;

alertsList.innerHTML = "";

if(data.length === 0){

alertsList.innerHTML = "<li>No alerts detected</li>";
return;

}

data.forEach(alert => {

const li = document.createElement("li");

li.innerText = alert;

alertsList.appendChild(li);

/* popup alert */

if(alert.includes("SUSPICIOUS")){
alert("⚠ Suspicious Webcam Activity Detected!");
}

});

})
.catch(err => console.log("Alert server not running"));

}

loadLogs();
loadAlerts();

setInterval(loadLogs, 3000);
setInterval(loadAlerts, 4000);

});