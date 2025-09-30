{ pkgs, lib, config, inputs, ... }:

{
dotenv.enable = true;

  env={
    GREET = "devenv";
    };

  packages = with pkgs; [
    cmatrix
  ];

  languages = {
   nix.enable = true;
   python={
     enable = true;
     venv = {
       enable = true;
       quiet = true;
       requirements = ''
          dash
          pandas
          jinja2
          fastapi
          llm-adapters
          dash-bootstrap-components
          uvicorn[standard]
         '';
     };
   uv={
     enable = true;
     };
     };
    };

   # processes={
   #   server.exec = ''python app.py'';
   #   };

  # services.postgres.enable = true;

  scripts={
    app.exec = "python app.py";
    hello.exec = ''
    echo hello from $GREET
  '';
    };

  enterShell = ''
    hello         # Run scripts directly
    git --version # Use packages
  '';

  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';
}
