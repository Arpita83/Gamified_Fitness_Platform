// document.getElementById("submit").addEventListener("click", bmi);

// function bmi() {
//     var cm = parseInt(document.getElementById("cm").value);
//     var kg = parseFloat(document.getElementById("kg").value);
//     var bm;
//     var newCm = parseFloat(cm / 100);

//     bm = kg / (newCm * newCm);
//     bm = bm.toFixed(1);
//     console.log(bm);

//     document.getElementById("result").innerHTML = bm; 
// }



document.getElementById("submit").addEventListener("click", bmi);

function bmi() {
    var cm = parseInt(document.getElementById("cm").value);
    var kg = parseFloat(document.getElementById("kg").value);
    var bm;
    var newCm = parseFloat(cm / 100);

    bm = kg / (newCm * newCm);
    bm = bm.toFixed(1);
    console.log(bm);

    var resultElement = document.getElementById("result");
    resultElement.innerHTML = bm;

    if (bm <= 18.5) {
        resultElement.style.color = "black";
    }
    else if (bm > 18.5 &&bm <= 24.9) {
        resultElement.style.color = "blue";
    } else if (bm > 24.9 && bm <= 29.9) {
        resultElement.style.color = "green";
    }else if (bm > 24.9 && bm <= 39.9) {
        resultElement.style.color = "yellow";
    }else{
        resultElement.style.color = "red";
    }
}
