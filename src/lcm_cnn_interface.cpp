#include "src/lcm_cnn_interface.hpp"
#include "src/contact_estimation.hpp"
#include <stdlib.h>
#include <typeinfo>
// #include "src/config.hpp"

int main(int argc, char **argv)
{
    /// LCM: subscribe to channels:
    lcm::LCM lcm;
    if (!lcm.good())
        return 1;
    
    char resolved_path[PATH_MAX];
    realpath("../", resolved_path);
    std::cout << resolved_path << std::endl;
    YAML::Node config_ = YAML::LoadFile(std::string(resolved_path) + "/config/interface.yaml");

    LcmMsgQueues_t lcm_msg_in;
    std::mutex mtx;
    LcmHandler handlerObject(&lcm, &lcm_msg_in, &mtx);
    // lcm.subscribe("leg_control_data", &Handler::receiveLegControlMsg, &handlerObject);
    // lcm.subscribe("microstrain", &Handler::receiveMicrostrainMsg, &handlerObject);
    // lcm.subscribe("contact_ground_truth", &Handler::receiveContactGroundTruthMsg, &handlerObject);

    std::cout << "Start Running LCM-CNN Interface" << std::endl;

    // Takes input arguments
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        // samplesCommon::printHelpInfo();
        return -1;
    }
    if (args.help)
    {
        // samplesCommon::printHelpInfo();
        return -1;
    }

    /// INTERFACE: use multiple threads to avoid missing messages:
    std::queue<float *> cnn_input_queue;
    std::queue<float *> new_data_queue;

    std::ofstream myfile;
    std::ofstream myfile_leg_p;
    // std::string PROGRAM_PATH = "/media/jetson256g/code/LCM_CNN_INTERFACE/deep-contact-estimator/";
    int debug_flag = config_["debug_flag"].as<int>();
    std::cout << "debug_flag: " << debug_flag << std::endl;
    std::string PROGRAM_PATH = config_["program_path"].as<std::string>();
    int input_w = config_["input_w"].as<int>();
    int input_h = config_["input_h"].as<int>();
    int num_legs = config_["num_legs"].as<int>();


    if (debug_flag == 1)
    {
       myfile.open(PROGRAM_PATH + "contact_est_lcm.csv");
       myfile_leg_p.open(PROGRAM_PATH + "p_lcm.csv");
    }

    LcmCnnInterface matrix_builder(args, &lcm_msg_in, &mtx, debug_flag, myfile_leg_p, &config_, input_h, input_w);
    ContactEstimation engine_builder(args, &lcm, &mtx, debug_flag, myfile, myfile_leg_p, &lcm_msg_in, &config_, input_h, input_w, num_legs);
    std::thread BuildMatrixThread(&LcmCnnInterface::buildMatrix, &matrix_builder, std::ref(cnn_input_queue), std::ref(new_data_queue));
    std::thread CNNInferenceThread(&ContactEstimation::makeInference, &engine_builder, std::ref(cnn_input_queue), std::ref(new_data_queue));

    std::cout << "started thread" << std::endl;

    while (0 == lcm.handle());
    
    BuildMatrixThread.join();
    CNNInferenceThread.join();

    if (debug_flag == 1) {
        myfile.close();
        myfile_leg_p.close();
    }

    return 0;
}

LcmCnnInterface::LcmCnnInterface(const samplesCommon::Args &args, LcmMsgQueues_t* lcm_msg_in, std::mutex* mtx, int debug_flag, std::ofstream& myfile_leg_p, 
                                YAML::Node* config, const int input_h, const int input_w)
    : config_(config),
      input_h_(input_h),
      input_w_(input_w),
      data_require(input_h),
      new_line(input_w, 0),
      sum_of_rows(input_w, 0),
      sum_of_rows_square(input_w, 0),
      previous_first_row(input_w, 0),
      cnn_input_matrix(input_h, std::vector<float>(input_w)),
      mean_vector(input_w, 0),
      std_vector(input_w, 0),
      is_first_full_matrix(true),
      lcm_msg_in_(lcm_msg_in),
      mtx_(mtx),
      debug_flag_(debug_flag),
      myfile_leg_p_(myfile_leg_p)
{
    q_dim = (*config_)["leg_q_dimension"].as<int>();
    qd_dim = (*config_)["leg_qd_dimension"].as<int>();
    p_dim = (*config_)["leg_p_dimension"].as<int>();
    v_dim = (*config_)["leg_v_dimension"].as<int>();
    tau_est_dim = (*config_)["leg_tau_dimension"].as<int>();

    acc_dim = (*config_)["imu_acc_dimension"].as<int>();
    omega_dim = (*config_)["imu_omega_dimension"].as<int>();
    quat_dim = (*config_)["imu_quat_dimension"].as<int>();
    rpy_dim = (*config_)["imu_rpy_dimension"].as<int>();
}

LcmCnnInterface::~LcmCnnInterface()
{
    /* 
    while (!cnn_input_leg_queue.empty())
    {
        delete[] cnn_input_leg_queue.front();
        cnn_input_leg_queue.pop();
    }

    while (!cnn_input_imu_queue.empty())
    {
        delete[] cnn_input_imu_queue.front();
        cnn_input_imu_queue.pop();
    }
    */
}

void LcmCnnInterface::buildMatrix(std::queue<float *> &cnn_input_queue, std::queue<float *> &new_data_queue)
{
    // Get leg input from queue
    while (true)
    {
        if (!lcm_msg_in_->cnn_input_leg_queue.empty() && !lcm_msg_in_->cnn_input_imu_queue.empty())
        {
            // mtx.lock();
            // // Get GTlabel from queue
            // int gtLabel = cnn_input_gtlabel_queue.front();
            // cnn_input_gtlabel_queue.pop();
            // mtx.unlock();

            // Start to build a new line and generate a new input
            int idx = 0; //!< keep track of the current new_line idx;
            // new_data stores the pointer to the new line
            float *new_data = new float[input_w_]();

            // get input data:
            std::shared_ptr<synced_proprioceptive_lcmt> synced_msgs = std::make_shared<synced_proprioceptive_lcmt>();

            mtx_->lock();
            std::shared_ptr<LcmLegStruct> leg_control_data = lcm_msg_in_->cnn_input_leg_queue.front();
            std::shared_ptr<LcmIMUStruct> microstrain_data = lcm_msg_in_->cnn_input_imu_queue.front();
            synced_msgs.get()->timestamp = lcm_msg_in_->timestamp_queue.front();
            lcm_msg_in_->timestamp_queue.pop();
            lcm_msg_in_->cnn_input_leg_queue.pop();
            lcm_msg_in_->cnn_input_imu_queue.pop();
            mtx_->unlock();


            // leg_control_data.q
            for (int i = 0; i < q_dim; ++i)
            {
                new_line[idx] = leg_control_data.get()->q[i];
                // new_data[idx] = new_line[idx];
                synced_msgs.get()->q[i] = leg_control_data.get()->q[i];
                ++idx;
            }
            // leg_control_data.qd:
            for (int i = 0; i < qd_dim; ++i)
            {
                new_line[idx] = leg_control_data.get()->qd[i];
                // new_data[idx] = new_line[idx];
                synced_msgs.get()->qd[i] = leg_control_data.get()->qd[i];
                ++idx;
            }
            // microstrain(IMU).acc:
            for (int i = 0; i < acc_dim; ++i)
            {
                new_line[idx] = microstrain_data.get()->acc[i];
                // new_data[idx] = new_line[idx];
                synced_msgs.get()->acc[i] = microstrain_data.get()->acc[i];
                ++idx;
            }
            // microstrain(IMU).omega:
            for (int i = 0; i < omega_dim; ++i)
            {
                new_line[idx] = microstrain_data.get()->omega[i];
                // new_data[idx] = new_line[idx]; 
                synced_msgs.get()->omega[i] = microstrain_data.get()->omega[i];
                ++idx;
            }
            // leg_control_data.p:
            for (int i = 0; i < p_dim; ++i)
            {
                new_line[idx] = leg_control_data.get()->p[i];
                // new_data[idx] = new_line[idx];
                synced_msgs.get()->p[i] = leg_control_data.get()->p[i];
                if (debug_flag_ == 1) {
                    myfile_leg_p_ << synced_msgs.get()->p[i] << ',';
                }
                ++idx;
            }

            if (debug_flag_ == 1)
            {
                myfile_leg_p_ << '\n';
                myfile_leg_p_ << std::flush;
            }

            // leg_control_data.v:
            for (int i = 0; i < v_dim; ++i)
            {
                new_line[idx] = leg_control_data.get()->v[i];
                // new_data[idx] = new_line[idx];
                synced_msgs.get()->v[i] = leg_control_data.get()->v[i];
                ++idx;
            }

            // leg_control_data.tau_est:
            for (int i = 0; i < tau_est_dim; ++i)
            {
                synced_msgs.get()->tau_est[i] = leg_control_data.get()->tau_est[i];
            }

            // microstrain.quat:
            for (int i = 0; i < quat_dim; ++i) 
            {
                synced_msgs.get()->quat[i] = microstrain_data.get()->quat[i];
            }

            // microstrain.rpy:
            for (int i = 0; i < rpy_dim; ++i) 
            {
                synced_msgs.get()->rpy[i] = microstrain_data.get()->rpy[i];
            }

            synced_msgs.get()->good_packets = microstrain_data.get()->good_packets;
            synced_msgs.get()->bad_packets = microstrain_data.get()->bad_packets;
            
            lcm_msg_in_->synced_msgs_queue.push(synced_msgs);
            // new_data_queue.push(new_data);

            // Put the new_line to the InputMatrix and destroy the first line:
            cnn_input_matrix.erase(cnn_input_matrix.begin());
            cnn_input_matrix.push_back(new_line);
            data_require = std::max(data_require - 1, 0);

            if (data_require != 0)
            {
                // delete[] new_data_queue.front();
                // new_data_queue.pop();
                lcm_msg_in_->synced_msgs_queue.pop();
            }
            else if (data_require == 0)
            {
                normalizeAndInfer(cnn_input_queue);
                // new_data_queue.push(new_data);
                // std::cout << "The ground truth label is: " << gtLabel << std::endl;
                // break;
            }
        }
    }
}

void LcmCnnInterface::normalizeAndInfer(std::queue<float *> &cnn_input_queue)
{
    /// REMARK: normalize input and send to CNN network in TRT
    // We need to normalize the input matrix, to do so,
    // we need to calculate the mean value and standard deviation.
    if (is_first_full_matrix)
    {
        runFullCalculation(cnn_input_queue);
        is_first_full_matrix = false;
    }
    else
    {
        runSlidingWindow(cnn_input_queue);
    }
}

void LcmCnnInterface::runFullCalculation(std::queue<float *> &cnn_input_queue)
{
    float *cnn_input_matrix_normalized = new float[input_h_ * input_w_]();

    // std::ofstream data_file;
    // data_file.open("input_matrix_500Hz.bin", ios::out | ios::binary);
    // if (!data_file)
    // {
    //     std::cerr << " Cannot open the file!" << std::endl;
    //     return;
    // }

    for (int j = 0; j < input_w_; ++j)
    {
        // find mean:
        for (int i = 0; i < input_h_; ++i)
        {
            sum_of_rows[j] += cnn_input_matrix[i][j];
            sum_of_rows_square[j] += std::pow(cnn_input_matrix[i][j], 2.0);
        }
        mean_vector[j] = sum_of_rows[j] / input_h_;

        // find std:
        for (int i = 0; i < input_h_; ++i)
        {
            std_vector[j] += std::pow((cnn_input_matrix[i][j] - mean_vector[j]), 2.0);
        }
        std_vector[j] = std::sqrt(std_vector[j] / (input_h_ - 1));

        // Normalize the matrix:
        for (int i = 0; i < input_h_; ++i)
        {
            cnn_input_matrix_normalized[i * input_w_ + j] = (cnn_input_matrix[i][j] - mean_vector[j]) / std_vector[j];
        }
        previous_first_row[j] = cnn_input_matrix[0][j];
    }

    /// REMARK: delete the following lines in actual use:
    // data_file.write(reinterpret_cast<char *>(&cnn_input_matrix_normalized[0]), input_h_ * input_w_ * sizeof(float));
    // data_file.close();
    cnn_input_queue.push(cnn_input_matrix_normalized);
}

void LcmCnnInterface::runSlidingWindow(std::queue<float *> &cnn_input_queue)
{
    float *cnn_input_matrix_normalized = new float[input_h_ * input_w_]();

    for (int j = 0; j < input_w_; ++j)
    {
        // find mean:
        sum_of_rows[j] = sum_of_rows[j] - previous_first_row[j] + new_line[j];
        sum_of_rows_square[j] = sum_of_rows_square[j] - std::pow(previous_first_row[j], 2.0) + std::pow(new_line[j], 2.0);

        mean_vector[j] = sum_of_rows[j] / input_h_;

        // find std:
        std_vector[j] = sum_of_rows_square[j] - 2 * mean_vector[j] * sum_of_rows[j] + input_h_ * std::pow(mean_vector[j], 2.0);
        std_vector[j] = std::sqrt(std_vector[j] / (input_h_ - 1));

        // Normalize the matrix:
        for (int i = 0; i < input_h_; ++i)
        {
            cnn_input_matrix_normalized[i * input_w_ + j] = (cnn_input_matrix[i][j] - mean_vector[j]) / std_vector[j];
        }
        previous_first_row[j] = cnn_input_matrix[0][j];
    }

    cnn_input_queue.push(cnn_input_matrix_normalized);
}

