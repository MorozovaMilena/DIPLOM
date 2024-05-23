$(document).ready(function() {
    $('#news-form').on('submit', function(e) {
        e.preventDefault();
        var newsText = $('#news-text').val();
        
        if (newsText.trim() === '') {
            alert('Please enter news text.');
            return;
        }
        
        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ text: newsText }),
            success: function(response) {
                var resultHtml = `<p>${response.result} with probability ${response.probability}</p>`;
                $('#result').html(resultHtml);
            },
            error: function() {
                alert('Error processing your request.');
            }
        });
    });
});