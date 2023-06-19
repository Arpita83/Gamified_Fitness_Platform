document.getElementById("submit").addEventListener("click", bmi);

function bmi() {
    var cm = parseInt(document.getElementById("cm").value);
    var kg = parseFloat(document.getElementById("kg").value);
    var bm;
    var newCm = parseFloat(cm / 100);

    bm = kg / (newCm * newCm);
    bm = bm.toFixed(1);
    console.log(bm);

    document.getElementById("result").innerHTML = bm; 
}