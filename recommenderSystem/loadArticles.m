function articles = loadArticles()

fid = fopen('articleNames.txt');

% Number of articles 
n = 60;

articles = cell(n, 1);
for i = 1:n
    line = fgets(fid);    
    articles{i} = strtrim(line);
end
fclose(fid);

endfunction
