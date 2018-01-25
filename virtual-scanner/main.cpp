#include <iostream>
#include <string>
#include <time.h>
#include <getopt.h>
#include "VirtualScanner.h"
using namespace std;

const int DEFAULT_VIEWS = 6;


typedef struct {int viewsNum; bool flipNormals; std::vector<string> fileNames;} Args;

void printHelp() {
    std::cout <<
		"Usage: virtual-scanner [OPTIONS] [FILE]\n"
		"Convert .obj/.off FILEs into `points` format and calculates the normals.\n"
            "--views <n>, -v:     The number of views for scanning. suggested value: 14, default: 6\n"
            "--flip, -n:          Flip normals orientation. Deault: off\n"
            "--help:              Show this help\n";
    exit(1);
}

Args processArgs(int argc, char** argv) {
	const char* const short_opts = "nv:";
    const option long_opts[] = {
            {"flip", 0, nullptr, 'n'},
            {"views", 0, nullptr, 'v'},
            {"help", 0, nullptr, 'h'},
            {nullptr, 0, nullptr, 0}
	};

	int viewNum = DEFAULT_VIEWS; // scanning view number

	bool flipNormals = false; // output normal flipping flag

    while (true) {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

        if (-1 == opt)
            break;

        switch (opt) {
        case 'n':
            flipNormals = true;
            break;

        case 'v':
            viewNum = std::stoi(optarg);
            break;

        case 'h': // -h or --help
        case '?': // Unrecognized option
        default:
            printHelp();
            break;
        }
	}

	std::vector<string> fileNames;
	for (int index = optind; index < argc; index++) {
		fileNames.push_back(argv[index]);
	}

	return {viewNum, flipNormals, fileNames};
}

int main(int argc, char* argv[])
{
	Args args = processArgs(argc, argv);

	vector<string> allFiles = args.fileNames;
	if(allFiles.size() < 1) {
		std::cout << "Needs at least one file.";
		exit(1);
	}

	//#pragma omp parallel for
	for (int i = 0; i < allFiles.size(); i++)
	{
		clock_t t1 = clock();
		VirtualScanner scanner;
		scanner.scanning(allFiles[i], args.viewsNum, args.flipNormals);
		clock_t t2 = clock();

		string messg = allFiles[i].substr(allFiles[i].rfind('\\') + 1) +
			" done! Time: " + to_string(t2 - t1) + "\n";
		cout << messg;
	}

	return 0;
}
