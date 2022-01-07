#include <iostream>
#include <vector>
using namespace std;

int main(int argc,char **argv){
	if(argc == 2){
		cout << "Codeforces Tool (cf) v1.0.0" << endl;
	}else{
		vector<string> args(argv, argv + argc);
		args[0]="/Users/siddharthsingh/Documents/Coding/advanced_cf_tool";
		string command = "";
		for(auto x:args){
			command += x + " ";
		}
		command.pop_back();
		system(command.c_str());
	}
}