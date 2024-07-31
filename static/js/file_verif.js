

document.addEventListener('DOMContentLoaded', function() {
    var dropZone = document.getElementById('drop-zone');
    var fileInput = document.getElementById('file');
    var uploadForm = document.getElementById('upload-form');

    // Highlight drop zone on drag over
    dropZone.addEventListener('dragover', function(event) {
        event.preventDefault();
        event.stopPropagation();
        dropZone.classList.add('dragover');
    });

    // Remove highlight on drag leave
    dropZone.addEventListener('dragleave', function(event) {
        event.preventDefault();
        event.stopPropagation();
        dropZone.classList.remove('dragover');
    });

    // Handle file drop
    dropZone.addEventListener('drop', function(event) {
        event.preventDefault();
        event.stopPropagation();
        dropZone.classList.remove('dragover');

        var files = event.dataTransfer.files;
        fileInput.files = files;
        validateFiles(files);
    });

    // Handle file input change
    fileInput.addEventListener('change', function() {
        var files = fileInput.files;
        validateFiles(files);
    });

    // Form submission
    uploadForm.addEventListener('submit', function(event) {
        var files = fileInput.files;
        if (!validateFiles(files)) {
            event.preventDefault();
        }
    });

    function validateFiles(files) {
        var allowedExtensions = ['png', 'jpg', 'jpeg', 'bmp'];
        var maxFiles = 999; // Maximum number of files allowed
        var valid = true;
        var errorMessage = "";

        if (files.length > maxFiles) {
            valid = false;
            errorMessage += "You can only upload up to " + maxFiles + " files.\n";
        }

        for (var i = 0; i < files.length; i++) {
            var file = files[i];
            var fileExtension = file.name.split('.').pop().toLowerCase();
            if (!allowedExtensions.includes(fileExtension)) {
                valid = false;
                errorMessage += "File " + file.name + " has an invalid extension.\n";
            }
        }

        if (!valid) {
            alert(errorMessage);
        }

        return valid;
    }
});