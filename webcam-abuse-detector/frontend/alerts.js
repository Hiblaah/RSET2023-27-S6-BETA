document.addEventListener("DOMContentLoaded", function(){

fetch("/alerts")
.then(res => res.json())
.then(data => {

const table = document.getElementById("alertTable");

if(!table) return;

table.innerHTML = "";

data.forEach(alert => {

const row = document.createElement("tr");

row.innerHTML = `
<td>${alert.application}</td>
<td>${alert.date}</td>
<td>${alert.time}</td>
<td>${alert.duration}</td>
<td style="color:red;font-weight:bold">${alert.status}</td>
`;

table.appendChild(row);

});

})
.catch(err => console.log("Error loading alerts:", err));

});