; ModuleID = 'first_simd.cpp'
source_filename = "first_simd.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%"struct.std::_Vector_base<double, std::allocator<double>>::_Vector_impl_data" = type { ptr, ptr, ptr }

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local void @_Z17memory_bound_simdRKSt6vectorIdSaIdEERS1_(ptr nocapture noundef nonnull readonly align 8 dereferenceable(24) %0, ptr nocapture noundef nonnull readonly align 8 dereferenceable(24) %1) local_unnamed_addr #0 {
  %3 = getelementptr inbounds %"struct.std::_Vector_base<double, std::allocator<double>>::_Vector_impl_data", ptr %0, i64 0, i32 1
  %4 = load ptr, ptr %3, align 8, !tbaa !5
  %5 = load ptr, ptr %0, align 8, !tbaa !10
  %6 = icmp eq ptr %4, %5
  br i1 %6, label %61, label %7

7:                                                ; preds = %2
  %8 = ptrtoint ptr %4 to i64
  %9 = ptrtoint ptr %5 to i64
  %10 = sub i64 %8, %9
  %11 = ashr exact i64 %10, 3
  %12 = load ptr, ptr %1, align 8, !tbaa !10
  %13 = tail call i64 @llvm.umax.i64(i64 %11, i64 1)
  %14 = icmp ult i64 %11, 16
  %15 = ptrtoint ptr %12 to i64
  %16 = sub i64 %15, %9
  %17 = icmp ult i64 %16, 128
  %18 = select i1 %14, i1 true, i1 %17
  br i1 %18, label %43, label %19

19:                                               ; preds = %7
  %20 = and i64 %13, -16
  br label %21

21:                                               ; preds = %21, %19
  %22 = phi i64 [ 0, %19 ], [ %39, %21 ]
  %23 = getelementptr inbounds double, ptr %5, i64 %22
  %24 = getelementptr inbounds double, ptr %23, i64 4
  %25 = getelementptr inbounds double, ptr %23, i64 8
  %26 = getelementptr inbounds double, ptr %23, i64 12
  %27 = load <4 x double>, ptr %23, align 8, !tbaa !11
  %28 = load <4 x double>, ptr %24, align 8, !tbaa !11
  %29 = load <4 x double>, ptr %25, align 8, !tbaa !11
  %30 = load <4 x double>, ptr %26, align 8, !tbaa !11
  %31 = fmul <4 x double> %27, %27
  %32 = fmul <4 x double> %28, %28
  %33 = fmul <4 x double> %29, %29
  %34 = fmul <4 x double> %30, %30
  %35 = getelementptr inbounds double, ptr %12, i64 %22
  %36 = getelementptr inbounds double, ptr %35, i64 4
  %37 = getelementptr inbounds double, ptr %35, i64 8
  %38 = getelementptr inbounds double, ptr %35, i64 12
  store <4 x double> %31, ptr %35, align 8, !tbaa !11
  store <4 x double> %32, ptr %36, align 8, !tbaa !11
  store <4 x double> %33, ptr %37, align 8, !tbaa !11
  store <4 x double> %34, ptr %38, align 8, !tbaa !11
  %39 = add nuw i64 %22, 16
  %40 = icmp eq i64 %39, %20
  br i1 %40, label %41, label %21, !llvm.loop !13

41:                                               ; preds = %21
  %42 = icmp eq i64 %13, %20
  br i1 %42, label %61, label %43

43:                                               ; preds = %7, %41
  %44 = phi i64 [ 0, %7 ], [ %20, %41 ]
  %45 = and i64 %13, 7
  %46 = icmp eq i64 %45, 0
  br i1 %46, label %57, label %47

47:                                               ; preds = %43, %47
  %48 = phi i64 [ %54, %47 ], [ %44, %43 ]
  %49 = phi i64 [ %55, %47 ], [ 0, %43 ]
  %50 = getelementptr inbounds double, ptr %5, i64 %48
  %51 = load double, ptr %50, align 8, !tbaa !11
  %52 = fmul double %51, %51
  %53 = getelementptr inbounds double, ptr %12, i64 %48
  store double %52, ptr %53, align 8, !tbaa !11
  %54 = add nuw i64 %48, 1
  %55 = add i64 %49, 1
  %56 = icmp eq i64 %55, %45
  br i1 %56, label %57, label %47, !llvm.loop !17

57:                                               ; preds = %47, %43
  %58 = phi i64 [ %44, %43 ], [ %54, %47 ]
  %59 = sub i64 %44, %13
  %60 = icmp ugt i64 %59, -8
  br i1 %60, label %61, label %62

61:                                               ; preds = %57, %62, %41, %2
  ret void

62:                                               ; preds = %57, %62
  %63 = phi i64 [ %103, %62 ], [ %58, %57 ]
  %64 = getelementptr inbounds double, ptr %5, i64 %63
  %65 = load double, ptr %64, align 8, !tbaa !11
  %66 = fmul double %65, %65
  %67 = getelementptr inbounds double, ptr %12, i64 %63
  store double %66, ptr %67, align 8, !tbaa !11
  %68 = add nuw i64 %63, 1
  %69 = getelementptr inbounds double, ptr %5, i64 %68
  %70 = load double, ptr %69, align 8, !tbaa !11
  %71 = fmul double %70, %70
  %72 = getelementptr inbounds double, ptr %12, i64 %68
  store double %71, ptr %72, align 8, !tbaa !11
  %73 = add nuw i64 %63, 2
  %74 = getelementptr inbounds double, ptr %5, i64 %73
  %75 = load double, ptr %74, align 8, !tbaa !11
  %76 = fmul double %75, %75
  %77 = getelementptr inbounds double, ptr %12, i64 %73
  store double %76, ptr %77, align 8, !tbaa !11
  %78 = add nuw i64 %63, 3
  %79 = getelementptr inbounds double, ptr %5, i64 %78
  %80 = load double, ptr %79, align 8, !tbaa !11
  %81 = fmul double %80, %80
  %82 = getelementptr inbounds double, ptr %12, i64 %78
  store double %81, ptr %82, align 8, !tbaa !11
  %83 = add nuw i64 %63, 4
  %84 = getelementptr inbounds double, ptr %5, i64 %83
  %85 = load double, ptr %84, align 8, !tbaa !11
  %86 = fmul double %85, %85
  %87 = getelementptr inbounds double, ptr %12, i64 %83
  store double %86, ptr %87, align 8, !tbaa !11
  %88 = add nuw i64 %63, 5
  %89 = getelementptr inbounds double, ptr %5, i64 %88
  %90 = load double, ptr %89, align 8, !tbaa !11
  %91 = fmul double %90, %90
  %92 = getelementptr inbounds double, ptr %12, i64 %88
  store double %91, ptr %92, align 8, !tbaa !11
  %93 = add nuw i64 %63, 6
  %94 = getelementptr inbounds double, ptr %5, i64 %93
  %95 = load double, ptr %94, align 8, !tbaa !11
  %96 = fmul double %95, %95
  %97 = getelementptr inbounds double, ptr %12, i64 %93
  store double %96, ptr %97, align 8, !tbaa !11
  %98 = add nuw i64 %63, 7
  %99 = getelementptr inbounds double, ptr %5, i64 %98
  %100 = load double, ptr %99, align 8, !tbaa !11
  %101 = fmul double %100, %100
  %102 = getelementptr inbounds double, ptr %12, i64 %98
  store double %101, ptr %102, align 8, !tbaa !11
  %103 = add nuw i64 %63, 8
  %104 = icmp eq i64 %11, %103
  br i1 %104, label %61, label %62, !llvm.loop !19
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="skylake" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+bmi,+bmi2,+clflushopt,+cmov,+crc32,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+invpcid,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prfchw,+rdrnd,+rdseed,+sahf,+sgx,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-amx-bf16,-amx-complex,-amx-fp16,-amx-int8,-amx-tile,-avx10.1-256,-avx10.1-512,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512fp16,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxifma,-avxneconvert,-avxvnni,-avxvnniint16,-avxvnniint8,-cldemote,-clwb,-clzero,-cmpccxadd,-enqcmd,-fma4,-gfni,-hreset,-kl,-lwp,-movdir64b,-movdiri,-mwaitx,-pconfig,-pku,-prefetchi,-prefetchwt1,-ptwrite,-raoint,-rdpid,-rdpru,-rtm,-serialize,-sha,-sha512,-shstk,-sm3,-sm4,-sse4a,-tbm,-tsxldtrk,-uintr,-usermsr,-vaes,-vpclmulqdq,-waitpkg,-wbnoinvd,-widekl,-xop" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"Ubuntu clang version 18.1.3 (1ubuntu1)"}
!5 = !{!6, !7, i64 8}
!6 = !{!"_ZTSNSt12_Vector_baseIdSaIdEE17_Vector_impl_dataE", !7, i64 0, !7, i64 8, !7, i64 16}
!7 = !{!"any pointer", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!6, !7, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"double", !8, i64 0}
!13 = distinct !{!13, !14, !15, !16}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"llvm.loop.isvectorized", i32 1}
!16 = !{!"llvm.loop.unroll.runtime.disable"}
!17 = distinct !{!17, !18}
!18 = !{!"llvm.loop.unroll.disable"}
!19 = distinct !{!19, !14, !15}
