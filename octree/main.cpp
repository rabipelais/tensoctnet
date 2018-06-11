﻿#define _USE_MATH_DEFINES
#include <cmath>
#include <float.h>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <getopt.h>
#include <Miniball.hpp>
#include "Octree.h"

using std::vector;
using std::string;
using std::cout;

void get_all_filenames(vector<string>& filenames, const string& data_list);
void load_pointcloud(vector<float>& pt, vector<float>& normal,
	vector<int>& seg, const string& _filename);
void rotation_matrix(float* rot, const float angle, const float* axis);
void matrix_prod(float* C, const float*A, const float*B,
	const int M, const int N, const int K);
//// The following function computes the smallest enclosing balls of the given
//// point cloud. It depends on the header Miniball.hpp which can be downloaded
//// from https://people.inf.ethz.ch/gaertner/subdir/software/miniball.html
void bounding_sphere(float& radius, float* center, const float* V, const int n);
void bounding_sphere_fast(float& radius, float* center, const float* V, const int n);

const int FULL_LAYER_DEFAULT = 2;
const float DISPLACEMENT_DEFAULT = 0.55;

void printHelp() {
    std::cout <<
		"Usage: octree [OPTION] [FILE]...\n"
		"Convert FILEs in POINTS format to Octrees.\n"
		"--depth <n>, -d: the maximum depth of the octree tree\n"
		"--full_layer <n>, -l: which layer of the octree is full. suggested value: 2\n"
		"--displacement <n>: the offset value for handing extremely thin shapes: suggested value: 0.55\n"
		"--segmentation, -s: a boolean value indicating whether the output is for the segmentation task.";

    exit(1);
}

typedef struct {int depth; bool fullLayers; float displacement; bool segmentation; std::vector<string> fileNames;} Args;

Args processArgs(int argc, char** argv) {
	const char* const short_opts = "nv:";
    const option long_opts[] = {
            {"depth", required_argument, nullptr, 'd'},
            {"full_layer", required_argument, nullptr, 'l'},
            {"displacement", required_argument, nullptr, 'p'},
            {"segmentation", no_argument, nullptr, 's'},
            {"help", 0, nullptr, 'h'},
            {nullptr, 0, nullptr, 0}
	};

	int depth = 4;
	bool depthSet = false;
	int fullLayer = FULL_LAYER_DEFAULT;
	float displacement = DISPLACEMENT_DEFAULT;
	bool segmentation = false;

    while (true) {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

        if (-1 == opt)
            break;

        switch (opt) {
        case 'd':
			depthSet = true;
			depth = std::stoi(optarg);
            break;

        case 'l':
			fullLayer = std::stoi(optarg);
            break;

        case 'p':
            displacement = std::stof(optarg);
            break;

        case 's':
            segmentation = true;
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

	if(!depthSet) {
		std::cout << "`DEPTH` is a required option." << std::endl;
		printHelp();
	}

	return {depth, fullLayer, displacement, segmentation, fileNames};
}

int main(int argc, char* argv[])
{
	Args args = processArgs(argc, argv);

	vector<string> allFiles = args.fileNames;
	if(allFiles.size() < 1) {
		std::cout << "Needs at least one file.\n";
		printHelp();
		exit(1);
	}

	int depth = args.depth;
	int full_layer = args.fullLayers;
	float dis = args.displacement;
	int view_num = 12; //TODO make arg?
	bool segmentation = args.segmentation;

	for (int i = 0; i < allFiles.size(); i++)
	{
		vector<float> Pt, normal, C;
		vector<int> seg;
		load_pointcloud(Pt, normal, seg, allFiles[i]);

		// bounding sphere
		float radius;
		float center[3];
		int npt = Pt.size() / 3;
		bounding_sphere(radius, center, Pt.data(), npt);
		//bounding_sphere_fast(radius, center, Pt.data(), npt);

		// centralize
		//Pt.colwise() -= center;
		for (int n = 0; n < npt; ++n)
		{
			for (int m = 0; m < 3; ++m)
			{
				Pt[3 * n + m] -= center[m];
			}
		}

		// displacement
		if (dis > 1.0e-10)
		{
			float D = dis * 2.0 * radius / float(1 << depth);
			//Pt += normal*D;
			for (int n = 0; n < npt; ++n)
			{
				for (int m = 0; m < 3; ++m)
				{
					Pt[3 * n + m] += normal[3 * n + m] * D;
				}
			}
			radius += D;
		}

		// data augmentation
		float bbmin[] = { -radius, -radius, -radius };
		float bbmax[] = { radius, radius, radius };
		// For ModelNet40, the upright direction is Z axis
		float axis[] = { 0.0f, 0.0f, 1.0f };
		// IMPORTANT: for ShapeNet55, the upright direction is Y axis
		// please un-comment the following line and rebuild the code.
		// axis[1] = 1.0f; axis[2] = 0.0f;
		float Rot[9];
		rotation_matrix(Rot, 2.0f*M_PI / float(view_num), axis);
		//Rot = Eigen::AngleAxis<float>(2.0*M_PI / float(view_num),
		//								Eigen::Vector3f(0, 0, 1));
		for (int v = 0; v < view_num; ++v)
		{
			Octree octree;
			octree.build(depth, full_layer, npt, bbmin, bbmax,
				Pt.data(), normal.data(), seg.data());

			// rotate point
			//Pt = Rot * Pt;
 			//normal = Rot * normal;
			C.resize(3 * npt);
			matrix_prod(C.data(), Rot, Pt.data(), 3, npt, 3);
			swap(C, Pt);
			matrix_prod(C.data(), Rot, normal.data(), 3, npt, 3);
			swap(C, normal);

			// save

			// message
			char file_suffix[128];
			sprintf(file_suffix, "_%d_%d_%03d.octree", depth, full_layer, v);
			string filename = allFiles[i].substr(0, allFiles[i].rfind('.')) + file_suffix;
			cout << "Processed: " << filename.substr(filename.rfind('\\') + 1) << std::endl;
			if(octree.save(filename)) {
			  cout << "-- saved to: " << filename << std::endl;
			} else {
			  cout << "-- error saving file: " << filename << std::endl;
			}
		}
	}

	return 0;
}

void get_all_filenames(vector<string>& filenames, const string& data_list)
{
	std::ifstream infile(data_list);
	string line;
	while (std::getline(infile, line))
	{
		filenames.push_back(line);
	}
	infile.close();

}

void load_pointcloud(vector<float>& pt,	vector<float>& normal,
	vector<int>& seg, const string& filename)
{
	std::ifstream infile(filename, std::ios::binary);

	infile.seekg(0, infile.end);
	int len = infile.tellg();
	infile.seekg(0, infile.beg);

	int n;
	infile.read((char*)(&n), sizeof(int));
	pt.resize(3 * n);
	infile.read((char*)pt.data(), sizeof(float)*3*n);
	normal.resize(3 * n);
	infile.read((char*)normal.data(), sizeof(float)*3*n);

	if (6 * n * sizeof(float) + (n + 1) * sizeof(int) == len)
	{
		seg.resize(n);
		infile.read((char*)seg.data(), sizeof(int)*n);
	}

	infile.close();
}

void bounding_sphere(float& radius, float* center, const float* V, const int n)
{
	int d = 3; // 3D mini-ball
	radius = center[0] = center[1] = center[2] = 0;
	if (n < 2) return;

	// mini-ball
	const float** ap = new const float*[n];
	for (int i = 0; i < n; ++i) { ap[i] = V + d * i; }
	typedef const float** PointIterator;
	typedef const float* CoordIterator;
	Miniball::Miniball <
		Miniball::CoordAccessor < PointIterator, CoordIterator >>
		miniball(d, ap, ap + n);

	// get result
	if (miniball.is_valid())
	{
		const float* cnt = miniball.center();
		for (int i = 0; i < d; ++i) {
			center[i] = cnt[i];
		}
		radius = sqrtf(miniball.squared_radius() + 1.0e-20f);
	}
	else
	{
		// the miniball might failed sometimes
		// if so, just calculate the bounding box

		float bbmin[3] = { V[0], V[1], V[2] };
		float bbmax[3] = { V[0], V[1], V[2] };
		for (int i = 1; i < n; ++i)
		{
			int i3 = i * 3;
			for (int j = 0; j < d; ++j)
			{
				float tmp = V[i3 + j];
				if (tmp < bbmin[j]) bbmin[j] = tmp;
				if (tmp > bbmax[j]) bbmax[j] = tmp;
			}
		}

		float width[3];
		for (int j = 0; j < d; ++j)
		{
			width[j] = (bbmax[j] - bbmin[j]) / 2.0f;
			center[j] = (bbmax[j] + bbmin[j]) / 2.0f;
		}

		radius = width[0];
		if (width[1] > radius) radius = width[1];
		if (width[2] > radius) radius = width[2];
	}

	// release
	delete[] ap;
}

void bounding_sphere_fast(float& radius, float* center, const float* V, const int n)
{
	float bb[3][2] = { { FLT_MAX,-FLT_MAX },{ FLT_MAX,-FLT_MAX },{ FLT_MAX,-FLT_MAX } };
	int id[6];
	for (int i = 0; i < 3 * n; i += 3)
	{
		if (V[i] < bb[0][0])
		{
			id[0] = i; bb[0][0] = V[i];
		}
		if (V[i] > bb[0][1])
		{
			id[1] = i; bb[0][1] = V[i];
		}
		if (V[i + 1] < bb[1][0])
		{
			id[2] = i; bb[1][0] = V[i + 1];
		}
		if (V[i + 1] > bb[1][1])
		{
			id[3] = i; bb[1][1] = V[i + 1];
		}
		if (V[i + 2] < bb[2][0])
		{
			id[4] = i; bb[2][0] = V[i + 2];
		}
		if (V[i + 2] > bb[2][1])
		{
			id[5] = i; bb[2][1] = V[i + 2];
		}
	}

	radius = 0;
	int choose_id = -1;
	for (int i = 0; i < 3; i++)
	{
		float dx = V[id[2 * i]] - V[id[2 * i + 1]];
		float dy = V[id[2 * i] + 1] - V[id[2 * i + 1] + 1];
		float dz = V[id[2 * i] + 2] - V[id[2 * i + 1] + 2];
		float r2 = dx * dx + dy * dy + dz * dz;
		if (r2 > radius)
		{
			radius = r2; choose_id = 2 * i;
		}
	}
	center[0] = 0.5f * (V[id[choose_id]] + V[id[choose_id + 1]]);
	center[1] = 0.5f * (V[id[choose_id] + 1] + V[id[choose_id + 1] + 1]);
	center[2] = 0.5f * (V[id[choose_id] + 2] + V[id[choose_id + 1] + 2]);

	float radius2 = radius * 0.25f;
	radius = sqrtf(radius2);

	for (int i = 0; i < 3 * n; i += 3)
	{
		float dx = V[i] - center[0], dy = V[i + 1] - center[1], dz = V[i + 2] - center[2];
		float dis2 = dx*dx + dy*dy + dz*dz;
		if (dis2 > radius2)
		{
			float old_to_p = sqrt(dis2);
			radius = (radius + old_to_p) * 0.5f;
			radius2 = radius  * radius;
			float old_to_new = old_to_p - radius;
			center[0] = (radius * center[0] + old_to_new * V[i]) / old_to_p;
			center[1] = (radius * center[1] + old_to_new * V[i + 1]) / old_to_p;
			center[2] = (radius * center[2] + old_to_new * V[i + 2]) / old_to_p;
		}
	}
}

void rotation_matrix(float* rot, const float angle, const float* axis)
{
	float cosa = cos(angle);
	float cosa1 = 1 - cosa;
	float sina = sin(angle);

	rot[0] = cosa + axis[0] * axis[0] * cosa1;
	rot[1] = axis[0] * axis[1] * cosa1 + axis[2] * sina;
	rot[2] = axis[0] * axis[2] * cosa1 - axis[1] * sina;

	rot[3] = axis[0] * axis[1] * cosa1 - axis[2] * sina;
	rot[4] = cosa + axis[1] * axis[1] * cosa1;
	rot[5] = axis[1] * axis[2] * cosa1 + axis[0] * sina;

	rot[6] = axis[0] * axis[2] * cosa1 + axis[1] * sina;
	rot[7] = axis[1] * axis[2] * cosa1 - axis[0] * sina;
	rot[8] = cosa + axis[2] * axis[2] * cosa1;
}


void matrix_prod(float* C, const float* A, const float* B,
	const int M, const int N, const int K)
{
	#pragma omp parallel for
	for (int n = 0; n < N; ++n)
	{
		for (int m = 0; m < M; ++m)
		{
			C[n*M + m] = 0;
			for (int k = 0; k < K; ++k)
			{
				C[n*M + m] += A[k*M + m] * B[n*K + k];
			}
		}
	}
}
