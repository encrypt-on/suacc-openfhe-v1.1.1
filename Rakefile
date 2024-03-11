require 'pathname'

project_root=File.dirname(__FILE__)
build_root="#{project_root}/bbuild"
active_builddir="#{build_root}/default"


cmd_project_info = [
    'git rev-parse --abbrev-ref HEAD', 'git --no-pager log -1 --pretty=format:"%h (%s) "', 'git status'
].join " && "

desc "contains tasks relating openfhe devel"
namespace :openfhe do

  desc "cmake configure"
  task :cmake_configure, [:current_directory] do |t, args|
    current_directory = args[:current_directory] || active_builddir
    abort "[Error] no CMakeLists.txt found" unless File.exists? "#{project_root}/CMakeLists.txt"

    Dir.chdir(current_directory) do
      rpath = Pathname.new(current_directory).relative_path_from(project_root)
      abort "[Error] '#{rpath}' didn't find **/bbuild/ " unless rpath.to_s =~ /(?:^|\/)bbuild(?:\/|$)/
      abort "[Error] '#{current_directory}' is not empty. Empty it first!" unless Dir.empty?(current_directory)
      cmake_opts = [ 
          "-DCMAKE_BUILD_TYPE=Debug", 
          "-DCMAKE_C_FLAGS_DEBUG=\"-g -O0 -Wno-maybe-uninitialized -Wno-unused-variable -Wno-unused-but-set-variable\"", 
          "-DCMAKE_CXX_FLAGS_DEBUG=\"-g -O0 -Wno-maybe-uninitialized -Wno-unused-variable -Wno-unused-but-set-variable\"", 
          "-DWITH_OPENMP=OFF", 
          "-DWITH_SU_GPU_NTT=ON", 
          "-DCMAKE_EXPORT_COMPILE_COMMANDS=1", 
          "-DCMAKE_INSTALL_PREFIX=#{current_directory}/installdir", 
          "#{project_root}" 
      ].join " "

      sh "{ echo #{cmake_opts}; } 2>&1 | tee #{current_directory}/bbuild-info.rake.log"
      sh "{ cmake #{cmake_opts}; } 2>&1 | tee #{current_directory}/cmake.rake.log"
    end
  end

  desc "make and install"
  task :make_install, [:current_directory] do |t, args|
    current_directory = args[:current_directory]  
    abort "[Error] Makefile not found" unless File.exist? "#{current_directory}/Makefile"
    # puts "continuing ..."
    Dir.chdir(current_directory) do
      sh "{ make -j12 && make install; } 2>&1 | tee #{current_directory}/make-install.rake.log"
    end
  end
  
end
