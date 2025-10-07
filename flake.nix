{
  description = "kernels-community tooling";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.follows = "hf-nix/nixpkgs";
    hf-nix.url = "github:huggingface/hf-nix";
  };

  outputs =
    {
      self,
      flake-utils,
      hf-nix,
      nixpkgs,
    }:
    let
      systems = with flake-utils.lib.system; [
        aarch64-darwin
        aarch64-linux
        x86_64-linux
      ];
    in
    flake-utils.lib.eachSystem systems (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        formatter = pkgs.nixfmt-tree;
      }
    );
}
