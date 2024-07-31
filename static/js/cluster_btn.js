
var btnPlus = document.getElementById('btn-plus');
var btnMoins = document.getElementById('btn-moins');
var inputValeur = document.getElementById('nb_cluster');

btnPlus.addEventListener('click', function() {
    var valeurActuelle = parseInt(inputValeur.value);
            inputValeur.value = valeurActuelle + 1;
});

btnMoins.addEventListener('click', function() {
    var valeurActuelle = parseInt(inputValeur.value);
    if (valeurActuelle > 1) { // Empêche de descendre en dessous de 1
        inputValeur.value = valeurActuelle - 1;
    }
});