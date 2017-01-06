#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <Eigen/Dense>
#include <flann/flann.hpp>

#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/Exporter.hpp>      // C++ exporter interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing fla



template <typename Type>
static Type DegToRad(Type angle_in_degrees)
{
	return angle_in_degrees * (Type)(M_PI / 180.0);
}

template <typename Type>
static Type RadToDeg(Type angle_in_radians)
{
	return angle_in_radians * (Type)(180.0 / M_PI);
}

//
//template<typename Type>
//static bool compute_rigid_transformation(
//	Eigen::Matrix<Type, Eigen::Dynamic, 3>& src,
//	Eigen::Matrix<Type, Eigen::Dynamic, 3>& dst,
//	Eigen::Matrix<Type, 4, 4>& mat)
//{
//	Eigen::Matrix<Type, 3, 3> R;
//	Eigen::Matrix<Type, 3, 1> t;
//	if (compute_rigid_transformation(src, dst, R, t))
//	{
//		mat.block(0, 0, 3, 3) = R;
//		mat.row(3).setZero();
//		mat.col(3) = t.homogeneous();
//		return true;
//	}
//	return false;
//}

#if 0
template<typename Type>
static bool compute_rigid_transformation(
	const std::vector<Eigen::Matrix<Type, 3, 1>>& src,
	const std::vector<Eigen::Matrix<Type, 3, 1>>& dst,
	Eigen::Matrix<Type, 3, 3>& R,
	Eigen::Matrix<Type, 3, 1>& t)
{
	//
	// Verify if the sizes of point arrays are the same 
	//
	assert(src.size() == dst.size());
	int pairSize = (int)src.size();
	Eigen::Matrix<Type, 3, 1> center_src(0, 0, 0), center_dst(0, 0, 0);

	// 
	// Compute centroid
	//
	for (int i = 0; i<pairSize; ++i)
	{
		center_src += src[i];
		center_dst += dst[i];
	}
	center_src /= (Type)pairSize;
	center_dst /= (Type)pairSize;


	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> S(pairSize, 3), D(pairSize, 3);
	for (int i = 0; i<pairSize; ++i)
	{
		S.row(i) = src[i] - center_src;
		D.row(i) = dst[i] - center_dst;
	}
	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> Dt = D.transpose();
	Eigen::Matrix<Type, 3, 3> H = Dt * S;
	Eigen::Matrix<Type, 3, 3> W, U, V;

	//
	// Compute SVD
	//
	Eigen::JacobiSVD<Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>> svd;
	svd.compute(H, Eigen::ComputeThinU | Eigen::ComputeThinV);

	if (!svd.computeU() || !svd.computeV())
	{
		//	std::cerr << "<Error> Decomposition error" << std::endl;
		return false;
	}

	//
	// Compute rotation matrix and translation vector
	// 
	Eigen::Matrix<Type, 3, 3> Vt = svd.matrixV().transpose();
	R = svd.matrixU() * Vt;
	t = center_dst - R * center_src;

	return true;
}
#endif


template<typename Type, int _Options = Eigen::RowMajor>
static bool compute_rigid_transformation(
	Eigen::Matrix<Type, Eigen::Dynamic, 3, _Options>& src,
	Eigen::Matrix<Type, Eigen::Dynamic, 3, _Options>& dst,
	Eigen::Matrix<Type, 3, 3, _Options>& R,
	Eigen::Matrix<Type, 1, 3, _Options>& t)
{
	//
	// Verify if the sizes of point arrays are the same 
	//
	assert(src.rows() == dst.rows());
	size_t point_count = (int)src.rows();
	Eigen::Matrix<Type, 3, 1> center_src(0, 0, 0), center_dst(0, 0, 0);

	// 
	// Compute centroid
	//
	for (int i = 0; i<point_count; ++i)
	{
		center_src += src.row(i);
		center_dst += dst.row(i);
	}
	center_src /= (Type)point_count;
	center_dst /= (Type)point_count;


	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> S(point_count, 3), D(point_count, 3);
	for (int i = 0; i<point_count; ++i)
	{
		S.row(i) << src.row(i).x() - center_src.x(), src.row(i).y() - center_src.y(), src.row(i).z() - center_src.z();
		D.row(i) << dst.row(i).x() - center_dst.x(), dst.row(i).y() - center_dst.y(), dst.row(i).z() - center_dst.z();
	}

	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> Dt = D.transpose();
	Eigen::Matrix<Type, 3, 3> H = Dt * S;
	Eigen::Matrix<Type, 3, 3> W, U, V;

	//
	// Compute SVD
	//
	Eigen::JacobiSVD<Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>> svd;
	svd.compute(H, Eigen::ComputeThinU | Eigen::ComputeThinV);

	if (!svd.computeU() || !svd.computeV())
	{
		//	std::cerr << "<Error> Decomposition error" << std::endl;
		return false;
	}

	//
	// Compute rotation matrix and translation vector
	// 
	Eigen::Matrix<Type, 3, 3> Vt = svd.matrixV().transpose();
	R = svd.matrixU() * Vt;
	t = center_dst - R * center_src;

	return true;
}


template<typename Type, int _Options = Eigen::ColMajor>
static bool compute_rigid_transformation(
	Eigen::Matrix<Type, Eigen::Dynamic, 3, _Options>& src,
	Eigen::Matrix<Type, Eigen::Dynamic, 3, _Options>& dst,
	Eigen::Matrix<Type, 3, 3, _Options>& R,
	Eigen::Matrix<Type, 3, 1, _Options>& t)
{
	//
	// Verify if the sizes of point arrays are the same 
	//
	assert(src.rows() == dst.rows());
	size_t point_count = (int)src.rows();
	Eigen::Matrix<Type, 3, 1> center_src(0, 0, 0), center_dst(0, 0, 0);

	// 
	// Compute centroid
	//
	for (int i = 0; i<point_count; ++i)
	{
		center_src += src.row(i);
		center_dst += dst.row(i);
	}
	center_src /= (Type)point_count;
	center_dst /= (Type)point_count;


	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> S(point_count, 3), D(point_count, 3);
	for (int i = 0; i<point_count; ++i)
	{
		S.row(i) << src.row(i).x() - center_src.x(), src.row(i).y() - center_src.y(), src.row(i).z() - center_src.z();
		D.row(i) << dst.row(i).x() - center_dst.x(), dst.row(i).y() - center_dst.y(), dst.row(i).z() - center_dst.z();
	}

	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> Dt = D.transpose();
	Eigen::Matrix<Type, 3, 3> H = Dt * S;
	Eigen::Matrix<Type, 3, 3> W, U, V;

	//
	// Compute SVD
	//
	Eigen::JacobiSVD<Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>> svd;
	svd.compute(H, Eigen::ComputeThinU | Eigen::ComputeThinV);

	if (!svd.computeU() || !svd.computeV())
	{
		//	std::cerr << "<Error> Decomposition error" << std::endl;
		return false;
	}

	//
	// Compute rotation matrix and translation vector
	// 
	Eigen::Matrix<Type, 3, 3> Vt = svd.matrixV().transpose();
	R = svd.matrixU() * Vt;
	t = center_dst - R * center_src;

	return true;
}


template<typename Type, int _Rows, int _Cols, int _Options>
void copy_from_mesh(
	const aiMesh* mesh,
	Eigen::Matrix<Type, _Rows, _Cols, _Options>& vertices,
	Eigen::Matrix<Type, _Rows, _Cols, _Options>& normals)
{
	assert(vertices.rows() == mesh->mNumVertices && normals.rows() == mesh->mNumVertices);

	for (size_t i = 0; i < mesh->mNumVertices; ++i)
	{
		const aiVector3D pos = mesh->mVertices[i];
		vertices.row(i) << pos.x, pos.y, pos.z;

		const aiVector3D normal = mesh->mNormals[i];
		normals.row(i) << normal.x, normal.y, normal.z;
	}
}




template<typename Type, int _Rows, int _Cols, int _Options>
void copy_to_mesh(
	const Eigen::Matrix<Type, _Rows, _Cols, _Options>& vertices,
	const Eigen::Matrix<Type, _Rows, _Cols, _Options>& normals,
	aiMesh*& mesh)
{
	assert(vertices.rows() == mesh->mNumVertices && normals.rows() == mesh->mNumVertices);

	for (size_t i = 0; i < mesh->mNumVertices; ++i)
	{
		mesh->mVertices[i].x = vertices.row(i).x();
		mesh->mVertices[i].y = vertices.row(i).y();
		mesh->mVertices[i].z = vertices.row(i).z();

		mesh->mNormals[i].x = normals.row(i).x();
		mesh->mNormals[i].y = normals.row(i).y();
		mesh->mNormals[i].z = normals.row(i).z();
	}
}



template<typename Type>
void copy_from_mesh(
	const aiMesh* mesh,
	std::vector<Eigen::Matrix<Type, 3, 1>>& vertices,
	std::vector<Eigen::Matrix<Type, 3, 1>>& normals)
{
	vertices.clear();
	normals.clear();
	for (size_t i = 0; i < mesh->mNumVertices; ++i)
	{
		const aiVector3D pos = mesh->mVertices[i];
		vertices.push_back(Eigen::Matrix<Type, 3, 1>(pos.x, pos.y, pos.z));

		const aiVector3D normal = mesh->mNormals[i];
		normals.push_back(Eigen::Matrix<Type, 3, 1>(normal.x, normal.y, normal.z));
	}
}


template<typename Type>
void copy_to_mesh(
	const std::vector<Eigen::Matrix<Type, 3, 1>>& vertices,
	const std::vector<Eigen::Matrix<Type, 3, 1>>& normals,
	aiMesh*& mesh)
{
	for (size_t i = 0; i < mesh->mNumVertices; ++i)
	{
		aiVector3D& vertex = mesh->mVertices[i];
		memcpy(&vertex, vertices[i].data(), sizeof(Type) * 3);

		aiVector3D& normal = mesh->mNormals[i];
		memcpy(&normal, normals[i].data(), sizeof(Type) * 3);
	}
}




template<typename Type>
void apply_random_rotation(
	const std::vector<Eigen::Matrix<Type, 3, 1>>& in_vertices,
	const std::vector<Eigen::Matrix<Type, 3, 1>>& in_normals,
	std::vector<Eigen::Matrix<Type, 3, 1>>& rot_vertices,
	std::vector<Eigen::Matrix<Type, 3, 1>>& rot_normals)
{
	rot_vertices.clear();
	rot_normals.clear();

	assert(in_vertices.size() == in_normals.size());

	rot_vertices.resize(in_vertices.size());
	rot_normals.resize(in_normals.size());

	Eigen::Transform<Type, 3, Eigen::Affine> transform;
	transform.setIdentity();
	transform.rotate(Eigen::AngleAxis<Type>(DegToRad(Type(90)), Eigen::Matrix<Type, 3, 1>::Random()));

	for (std::size_t i = 0; i < in_vertices.size(); ++i)
	{
		const Eigen::Matrix<Type, 4, 1>& v = in_vertices[i].homogeneous();
		const Eigen::Matrix<Type, 4, 1>& n = in_normals[i].homogeneous();

		Eigen::Matrix<Type, 4, 1> rv = transform.matrix() * v;
		rv /= rv.w();

		Eigen::Matrix<Type, 4, 1> rn = transform.matrix() * n;
		rn /= rn.w();

		rot_vertices.at(i) = (rv.head<3>());
		rot_normals.at(i) = (rn.head<3>()).normalized();
	}

}

template<typename Type>
void apply_rotation(
	const std::vector<Eigen::Matrix<Type, 3, 1>>& in_vertices,
	const std::vector<Eigen::Matrix<Type, 3, 1>>& in_normals,
	Type x_angle,
	Type y_angle,
	Type z_angle,
	std::vector<Eigen::Matrix<Type, 3, 1>>& rot_vertices,
	std::vector<Eigen::Matrix<Type, 3, 1>>& rot_normals)
{
	rot_vertices.clear();
	rot_normals.clear();

	assert(in_vertices.size() == in_normals.size());

	rot_vertices.resize(in_vertices.size());
	rot_normals.resize(in_normals.size());

	Eigen::Transform<Type, 3, Eigen::Affine> transform;
	transform.setIdentity();
	transform.rotate(Eigen::AngleAxis<Type>(DegToRad(Type(x_angle)), Eigen::Matrix<Type, 3, 1>(1, 0, 0)));
	transform.rotate(Eigen::AngleAxis<Type>(DegToRad(Type(y_angle)), Eigen::Matrix<Type, 3, 1>(0, 1, 0)));
	transform.rotate(Eigen::AngleAxis<Type>(DegToRad(Type(z_angle)), Eigen::Matrix<Type, 3, 1>(0, 0, 1)));

	for (std::size_t i = 0; i < in_vertices.size(); ++i)
	{
		const Eigen::Matrix<Type, 4, 1>& v = in_vertices[i].homogeneous();
		const Eigen::Matrix<Type, 4, 1>& n = in_normals[i].homogeneous();

		Eigen::Matrix<Type, 4, 1> rv = transform.matrix() * v;
		rv /= rv.w();

		Eigen::Matrix<Type, 4, 1> rn = transform.matrix() * n;
		rn /= rn.w();

		rot_vertices.at(i) = (rv.head<3>());
		rot_normals.at(i) = (rn.head<3>()).normalized();
	}

}


template<typename Type>
void apply_transform(
	const std::vector<Eigen::Matrix<Type, 3, 1>>& in_vertices,
	const std::vector<Eigen::Matrix<Type, 3, 1>>& in_normals,
	const Eigen::Matrix<Type, 4, 4>& transform,
	std::vector<Eigen::Matrix<Type, 3, 1>>& out_vertices,
	std::vector<Eigen::Matrix<Type, 3, 1>>& out_normals)
{
	out_vertices.clear();
	out_normals.clear();

	assert(in_vertices.size() == in_normals.size());

	out_vertices.resize(in_vertices.size());
	out_normals.resize(in_normals.size());

	for (std::size_t i = 0; i < in_vertices.size(); ++i)
	{
		const Eigen::Matrix<Type, 4, 1>& v = in_vertices[i].homogeneous();
		const Eigen::Matrix<Type, 4, 1>& n = in_normals[i].homogeneous();

		Eigen::Matrix<Type, 4, 1> tv = transform.matrix() * v;
		tv /= tv.w();

		Eigen::Matrix<Type, 4, 1> tn = transform.matrix() * n;
		tn /= tn.w();

		out_vertices.at(i) = (tv.head<3>());
		out_normals.at(i) = (tn.head<3>()).normalized();
	}

}



template<typename Type, int _Rows, int _Cols, int _Options>
void apply_rotation(
	const Eigen::Matrix<Type, _Rows, _Cols, _Options>& in_vertices,
	const Eigen::Matrix<Type, _Rows, _Cols, _Options>& in_normals,
	Type x_angle,
	Type y_angle,
	Type z_angle,
	Eigen::Matrix<Type, _Rows, _Cols, _Options>& rot_vertices,
	Eigen::Matrix<Type, _Rows, _Cols, _Options>& rot_normals)
{
	assert(in_vertices.rows() == in_normals.rows() && in_vertices.cols() == in_normals.cols());

	Eigen::Transform<Type, _Cols, Eigen::Affine> transform;
	transform.setIdentity();
	transform.rotate(Eigen::AngleAxis<Type>(DegToRad(Type(x_angle)), Eigen::Matrix<Type, 3, 1>(1, 0, 0)));
	transform.rotate(Eigen::AngleAxis<Type>(DegToRad(Type(y_angle)), Eigen::Matrix<Type, 3, 1>(0, 1, 0)));
	transform.rotate(Eigen::AngleAxis<Type>(DegToRad(Type(z_angle)), Eigen::Matrix<Type, 3, 1>(0, 0, 1)));

	rot_vertices = in_vertices * transform.rotation();
	rot_normals = in_normals * transform.rotation();

	//std::cout
	//	<< "Input: " << std::endl
	//	<< in_vertices << std::endl
	//	<< "Output: " << std::endl
	//	<< rot_vertices << std::endl
	//	<< std::endl;
	std::cout
		<< std::endl
		<< "Transform Matrix : " << std::endl
		<< transform.matrix() << std::endl
		<< std::endl;
}

#define ASSIMP_DOUBLE_PRECISION
typedef ai_real Decimal;
int main(int argc, char* argv[])
{

	std::cout
		<< std::fixed << std::endl
		<< "Usage            : ./<app.exe> <input_model> <output_format> <rot_x> <rot_y> <rot_z>" << std::endl
		<< "Default          : ./icp.exe ../../data/cow.obj obj 10 25 15" << std::endl
		<< std::endl;

	const int Options = Eigen::RowMajor;
	const int Dimension = 3;
	const int NumNeighbours = 1;	// only the closest 
	const int KdTreeCount = 4;
	const int KnnSearchChecks = 128;

	std::string input_filename = "../../data/teddy.obj";
	std::string output_format = input_filename.substr(input_filename.size() - 3, 3);
	const Decimal x_rot = (Decimal)((argc > 3) ? atof(argv[3]) : 5);
	const Decimal y_rot = (Decimal)((argc > 4) ? atof(argv[4]) : 5);
	const Decimal z_rot = (Decimal)((argc > 5) ? atof(argv[5]) : 5);

	if (argc > 1)
		input_filename = argv[1];
	if (argc > 2)
		output_format = argv[2];



	//
	// Import file
	// 
	Assimp::Importer importer;
	const aiScene *scene = importer.ReadFile(input_filename, aiProcessPreset_TargetRealtime_Fast);//aiProcessPreset_TargetRealtime_Fast has the configs you'll need
	if (scene == nullptr)
	{
		std::cout << "Error: Could not read file: " << input_filename << std::endl;
		return EXIT_FAILURE;
	}


	//
	// Output info
	// 
	aiMesh *mesh = scene->mMeshes[0]; //assuming you only want the first mesh
	std::cout
		<< "Input File       : " << input_filename << std::endl
		<< "Vertices         : " << mesh->mNumVertices << std::endl
		<< "Faces            : " << mesh->mNumFaces << std::endl;


	Eigen::Matrix<Decimal, Eigen::Dynamic, 3, Options> in_vertices(mesh->mNumVertices, 3);
	Eigen::Matrix<Decimal, Eigen::Dynamic, 3, Options> in_normals(mesh->mNumVertices, 3);
	copy_from_mesh(mesh, in_vertices, in_normals);

	Eigen::Matrix<Decimal, Eigen::Dynamic, 3, Options> target_vertices(mesh->mNumVertices, 3);
	Eigen::Matrix<Decimal, Eigen::Dynamic, 3, Options> target_normals(mesh->mNumVertices, 3);
	apply_rotation(in_vertices, in_normals, x_rot, y_rot, z_rot, target_vertices, target_normals);
	copy_to_mesh(target_vertices, target_normals, mesh);



	//
	// Composing file name
	// 
	std::stringstream ss;
	ss << input_filename.substr(0, input_filename.size() - 4)
		<< "_rot_"
		<< x_rot << '_' << y_rot << '_' << z_rot
		<< '.' << output_format;
	std::string random_transf_filename = ss.str();

	//
	// Exporting input transformed vertices
	// 
	Assimp::Exporter exporter;
	aiReturn ret = exporter.Export(scene, output_format, random_transf_filename, scene->mFlags);
	if (ret == aiReturn_SUCCESS)
		std::cout << "Transformed file : " << random_transf_filename << std::endl;
	else
		std::cout << "Transformed file : <ERROR> file not saved - " << random_transf_filename << std::endl;



	const size_t vertex_array_count = mesh->mNumVertices * Dimension;
	Decimal* vertex_array_input = in_vertices.data();
	Decimal* vertex_array_query = target_vertices.data();


	// for each vertex find nearest neighbour
	const size_t NumInput = vertex_array_count / Dimension;
	const size_t NumQuery = NumInput;

	flann::Matrix<Decimal> dataset(vertex_array_input, NumInput, Dimension);
	flann::Matrix<Decimal> query(vertex_array_query, NumQuery, Dimension);

	flann::Matrix<int> indices(new int[query.rows * NumNeighbours], query.rows, NumNeighbours);
	flann::Matrix<Decimal> dists(new Decimal[query.rows * NumNeighbours], query.rows, NumNeighbours);

	// construct an randomized kd-tree index using 'KdTreeCount' kd-trees
	flann::Index<flann::L2<Decimal>> index(dataset, flann::KDTreeIndexParams(KdTreeCount));
	index.buildIndex();

	//std::cout << std::endl << in_vertices << std::endl << std::endl;

	// do a knn search, using 128 checks
	index.knnSearch(query, indices, dists, NumNeighbours, flann::SearchParams(KnnSearchChecks));	//flann::SearchParams(128));

	Eigen::Matrix<Decimal, Eigen::Dynamic, 3, Options> closest_neighbours(mesh->mNumVertices, 3);

	std::cout << "Indices: " << indices.rows << ", " << indices.cols << std::endl;

	int n = 0;
	for (int i = 0; i < indices.rows; ++i)	// loop through all vertices
	{
		const Decimal qx = query[i][0];
		const Decimal qy = query[i][1];
		const Decimal qz = query[i][2];

		for (int j = 0; j < indices.cols; ++j)	// loop through all neighbours found. [cols == NumNeighbours]
		{
			// resultant points
			const int index = indices[i][j];
			const Decimal x = static_cast<Decimal>(dataset[index][0]);
			const Decimal y = static_cast<Decimal>(dataset[index][1]);
			const Decimal z = static_cast<Decimal>(dataset[index][2]);

			closest_neighbours.row(i) << x, y, z;
		}

	}

	Eigen::Matrix<Decimal, Eigen::Dynamic, 3, Options> mat_target = target_vertices;
	Eigen::Matrix<Decimal, Eigen::Dynamic, 3, Options> mat_tmp = in_vertices;

	Eigen::Matrix<Decimal, 3, 3, Options> R = Eigen::Matrix<Decimal, 3, 3, Options>::Identity();
	Eigen::Matrix<Decimal, Options == Eigen::ColMajor ? 3 : 1, Options == Eigen::ColMajor ? 1 : 3, Options> t = Eigen::Matrix<Decimal, 3, 1, Options>::Zero();

	compute_rigid_transformation<Decimal, Options>(target_vertices, closest_neighbours, R, t);
	
	Eigen::Matrix<Decimal, 4, 4> transform = Eigen::Matrix<Decimal, 4, 4>::Zero();
	transform.block(0, 0, 3, 3) = R;
	transform.col(3) = t.homogeneous();

	std::cout
		<< std::endl
		<< "Transform Matrix : " << std::endl
		<< transform << std::endl << std::endl
		<< std::endl;


	//
	// Composing result file name
	// 
	std::stringstream result_ss;
	result_ss << input_filename.substr(0, input_filename.size() - 4)
		<< "_result_"
		<< x_rot << '_' << y_rot << '_' << z_rot
		<< '.' << output_format;
	std::string result_transf_filename = result_ss.str();
	
	Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic, Options> result_verts(in_vertices.rows(), 4);
	result_verts.block(0, 0, in_vertices.rows(), in_vertices.cols()) = in_vertices;
	result_verts.col(result_verts.cols() - 1).setOnes();
	result_verts = result_verts * transform;
	result_verts.conservativeResize(result_verts.rows(), result_verts.cols() - 1);
	copy_to_mesh(target_vertices, target_normals, mesh);

	//
	// Exporting input transformed vertices
	// 
	ret = exporter.Export(scene, output_format, result_transf_filename, scene->mFlags);
	if (ret == aiReturn_SUCCESS)
		std::cout << "Result file : " << result_transf_filename << std::endl;
	else
		std::cout << "Result file : <ERROR> file not saved - " << result_transf_filename << std::endl;

	std::cout << "==========================================" << std::endl;

}




