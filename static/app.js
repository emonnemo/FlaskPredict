function submitForm(formNode) {
	// var inputs = formNode.getElementsByTagName('input');
	// for (var i = 0; i < inputs.length; i++) {
	// 	if (inputs[i].value == "") {
	// 		alert('Field cannot be empty');
	// 		return false;
	// 	}
	// }
	var xhtr = new XMLHttpRequest();
	xhtr.onreadystatechange = function() {
		if (this.readyState == 4) {
			document.getElementById('predict-result').innerHTML = this.responseText;
		}

	}
	xhtr.open("POST","http://localhost:5000/predict", true);
	var formData = new FormData(formNode);
	xhtr.send(formData);
	return false;
}